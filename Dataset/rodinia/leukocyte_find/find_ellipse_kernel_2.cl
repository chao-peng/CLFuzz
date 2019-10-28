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
                           __constant float *c_cos_angle, __constant int *c_tX, __constant int *c_tY, __global float *gicov, int width, int height) {


	int i, j, k, n, x, y;
	int gid = get_global_id(0);
	if(gid>=width*height)
	  return;

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
	for (k = 0; k < NCIRCLES; k++) {
		// Variables used to compute the mean and variance
		//  of the gradients along the current stencil
		float sum = 0.f, M2 = 0.f, mean = 0.f;

		// Iterate across each sample point in the current stencil
		for (n = 0; n < NPOINTS; n++) {
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

		// Finish computing the mean
		mean = sum / ((float) NPOINTS);

		// Finish computing the variance
		float var = M2 / ((float) (NPOINTS - 1));

		// Keep track of the maximal GICOV value seen so far
		if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
	}

	// Store the maximal GICOV value
	gicov[(i * grad_m) + j] = max_GICOV;
}


// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
