/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* Calculate beta variables */
__kernel void calc_beta_dev( __global float *beta_d, /* dim = length x nstates */
		__global float *b_d,    /* dim = nsymbols x nstates */
		float scale_t,          /* current scaling value */
		int nstates,            /* number of states */
		int obs_t,              /* current observation */
		int t, __global int* ocl_kernel_branch_triggered_recorder)                  /* current time */
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nstates)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		beta_d[(t * nstates) + idx] = beta_d[((t + 1) * nstates) + idx] *
			b_d[(obs_t * nstates) + idx] / scale_t;
	}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
