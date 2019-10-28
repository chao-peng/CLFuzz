// #if defined(cl_amd_fp64) || defined(cl_khr_fp64)

// #if defined(cl_amd_fp64)
// #pragma OPENCL EXTENSION cl_amd_fp64 : enable
// #elif defined(cl_khr_fp64)
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #endif

#define SCALE_FACTOR 300

/** added this function. was missing in original float version.
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */


__kernel void sum_kernel(__global float* partial_sums, int Nparticles, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);

	if(i == 0)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		int x;
		float sum = 0;
		int num_blocks = ceil((float) Nparticles / (float) THREADS_PER_BLOCK);
		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (x = 0; x < num_blocks || (private_ocl_kernel_loop_boundary_not_reached[0] = false); x++) {
private_ocl_kernel_loop_iter_counter[0]++;

			sum += partial_sums[x];
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
		partial_sums[0] = sum;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
