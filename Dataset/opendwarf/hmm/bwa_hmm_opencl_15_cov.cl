/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* dot production naive version */
__kernel void s_dot_kernel_naive(int n,                        /* number of elements */
		__global float *paramA,       /* first vector A */
		int offsetA,                  /* offset of vector A */
		__global float *paramB,       /* second vector B */
		int offsetB,                  /* offset of vector B */
		__global float *partialSum_d, __global int* ocl_kernel_loop_recorder) /* memory space for partial sum */
{__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	unsigned int i;
	unsigned int tid = get_local_id(0);
	unsigned int totalThreads = get_num_groups(0) * SDOT_BLOCK_SIZE;
	unsigned int offset = SDOT_BLOCK_SIZE * get_group_id(0);

	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(i = offset + tid; i < n || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i += totalThreads)
	{
private_ocl_kernel_loop_iter_counter[0]++;

		partialSum_d[i] = paramA[offsetA + i] * paramB[offsetB + i];
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
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
