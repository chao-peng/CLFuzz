/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* Re-estimate Pi values */
__kernel void est_pi_dev(__global float *pi_d,    /* lth = nstates */
		__global float *alpha_d, /* dim = length x nstates */
		__global float *beta_d,  /* dim = length x nstates */
		float sum_ab,            /* sum dot production of alpha_d and beta_d */
		int nstates)             /* number of states */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nstates)
	{
		pi_d[idx] = alpha_d[idx] * beta_d[idx] / sum_ab;
	}
}
