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


__kernel void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, __constant float *c_strel,
                            __global float *img, __global float *dilated, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[10];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 10; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);



	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Determine this thread's location in the matrix
	int thread_id = get_global_id(0); //(blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id % img_m;
	int j = thread_id / img_m;

	if(j > img_n) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
return;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0f;

	#ifdef USE_IMAGE
	// Define the sampler for accessing the image
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	#endif

	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	// Lingjie Zhang modificated at 11/06/2015

    if (j < img_n){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

        private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (el_i = 0; el_i < strel_m || (private_ocl_kernel_loop_boundary_not_reached[0] = false); el_i++) {
private_ocl_kernel_loop_iter_counter[0]++;

	    	y = i - el_center_i + el_i;
	    	// Make sure we have not gone off the edge of the matrix
	    	if ( (y >= 0) && (y < img_m) ) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

	    		// Iterate across the structuring element in the other dimension
	    		private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (el_j = 0; el_j < strel_n || (private_ocl_kernel_loop_boundary_not_reached[1] = false); el_j++) {
private_ocl_kernel_loop_iter_counter[1]++;

	    			x = j - el_center_j + el_j;
	    			// Make sure we have not gone off the edge of the matrix
	    			//  and that the current structuring element value is not zero
	    			if ( (x >= 0) &&
	    				 (x < img_n) &&
	    				 (c_strel[(el_i * strel_n) + el_j] != 0) ) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

	    					// Determine if this is the maximal value seen so far
	    					#ifdef USE_IMAGE
	    					int2 addr = {y, x};
	    					float temp = read_imagef(img, sampler, addr).x;
	    					#else
	    					int addr = (x * img_m) + y;
	    					float temp = img[addr];
	    					#endif
	    					if (temp > max) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);
max = temp;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}

	    			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
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
	    	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
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

	    // Store the maximum value found
	    dilated[(i * img_n) + j] = max;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
    // end of Lingjie Zhang's modification
for (int update_recorder_i = 0; update_recorder_i < 10; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
