/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* Normalize B matrix */
__kernel void scale_b_dev(  __global float *b_d, /* dim = nsymbols x nstates */
		__global float *c_d, /* lth = nstates, matrix-vector product of b_d and ones_s_d */
		int nstates,         /* number of states */
		int nsymbols, __global int* ocl_kernel_branch_triggered_recorder)        /* number of symbols */
{__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	// unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

	if (idx < nstates && idy < nsymbols)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		if (b_d[(idy * nstates) + idx] == 0)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

			b_d[(idy * nstates) + idx] = 1e-10;
		}
		else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

			b_d[(idy * nstates) + idx] = b_d[(idy * nstates) + idx] / c_d[idx];
		}
	}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
