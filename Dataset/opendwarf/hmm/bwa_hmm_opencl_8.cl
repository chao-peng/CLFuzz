/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)


/* Sum next iteration of xi variables */
__kernel void calc_xi_dev(   __global float *xi_sum_d,  /* dim = nstates x nstates */
		__global float *a_d,       /* dim = nstates x nstates */
		__global float *b_d,       /* dim = nsymbols x nstates */
		__global float *alpha_d,   /* dim = length x nstates */
		__global float *beta_d,    /* dim = length x nstates */
		float sum_ab,              /* sum dot production of (alpha_d+t*nstates) and (beta_d+t*nstates) */
		int nstates,               /* number of states */
		int obs_t,                 /* next observation at t + 1 */
		int t)                     /* current time */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	// unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

	if (idx < nstates && idy < nstates)
	{
		xi_sum_d[(idy * nstates) + idx] += alpha_d[(t * nstates) + idy] *
			a_d[(idy * nstates) + idx] *
			b_d[(obs_t * nstates) + idx] *
			beta_d[((t+1) * nstates) + idx] /
			sum_ab;
	}
}
