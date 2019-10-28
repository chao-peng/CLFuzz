/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)


/* Sum next iteration of gamma variables */
__kernel void calc_gamma_dev(__global float *gamma_sum_d, /* lth = nstates */
		__global float *alpha_d,     /* dim = length x nstates */
		__global float *beta_d,      /* dim = length x nstates */
		int nstates,                 /* number of states */
		int t)                       /* current time */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nstates)
	{
		gamma_sum_d[idx] += alpha_d[(t * nstates) + idx] *
			beta_d[(t * nstates) + idx];
	}
}
