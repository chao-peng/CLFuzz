// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)



// Kernel to find the maximal GICOV value at each pixel of a
//  video frame, based on the input x- and y-gradient matrices

__kernel void GICOV_kernel(int grad_m, __global float *grad_x, __global float *grad_y, __constant float *c_sin_angle,
                           __constant float *c_cos_angle, __constant int *c_tX, __constant int *c_tY, __global float *gicov, int width, int height, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);



	int i, j, k, n, x, y;
	int gid = get_global_id(0);
	if(gid>=width*height)
	  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
return;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


	// Determine this thread's pixel
	i = gid/width + MAX_RAD + 2;
	j = gid%width + MAX_RAD + 2;

	// Initialize the maximal GICOV score to 0
	float max_GICOV = 0.f;

	#ifdef USE_IMAGE
	// Define the sampler for accessing the images
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	#endif

	// Iterate across each stencil
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (k = 0; k < NCIRCLES || (private_ocl_kernel_loop_boundary_not_reached[0] = false); k++) {
private_ocl_kernel_loop_iter_counter[0]++;

		// Variables used to compute the mean and variance
		//  of the gradients along the current stencil
		float sum = 0.f, M2 = 0.f, mean = 0.f;

		// Iterate across each sample point in the current stencil
		private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (n = 0; n < NPOINTS || (private_ocl_kernel_loop_boundary_not_reached[1] = false); n++) {
private_ocl_kernel_loop_iter_counter[1]++;

			// Determine the x- and y-coordinates of the current sample point
			y = j + c_tY[(k * NPOINTS) + n];
			x = i + c_tX[(k * NPOINTS) + n];

			// Compute the combined gradient value at the current sample point
			#ifdef USE_IMAGE
			int2 addr = {y, x};
			float p = read_imagef(grad_x, sampler, addr).x * c_cos_angle[n] +
			          read_imagef(grad_y, sampler, addr).x * c_sin_angle[n];
			#else
			int addr = x * grad_m + y;
			float p = grad_x[addr] * c_cos_angle[n] + grad_y[addr] * c_sin_angle[n];
			#endif

			// Update the running total
			sum += p;

			// Partially compute the variance
			float delta = p - mean;
			mean = mean + (delta / (float) (n + 1));
			M2 = M2 + (delta * (p - mean));
		}
if (private_ocl_kernel_loop_iter_counter[1] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 1);
}if (private_ocl_kernel_loop_iter_counter[1] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 2);
}if (private_ocl_kernel_loop_iter_counter[1] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[1]) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 8);
}

		// Finish computing the mean
		mean = sum / ((float) NPOINTS);

		// Finish computing the variance
		float var = M2 / ((float) (NPOINTS - 1));

		// Keep track of the maximal GICOV value seen so far
		if (((mean * mean) / var) > max_GICOV) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
max_GICOV = (mean * mean) / var;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

	}
if (private_ocl_kernel_loop_iter_counter[0] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 1);
}if (private_ocl_kernel_loop_iter_counter[0] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 2);
}if (private_ocl_kernel_loop_iter_counter[0] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[0]) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 8);
}

	// Store the maximal GICOV value
	gicov[(i * grad_m) + j] = max_GICOV;
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}


// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
