/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)

/* Accumulate B values */
__kernel void acc_b_dev( __global float *b_d,      /* dim = nsymbols x nstates */
		__global float *alpha_d,  /* dim = length x nstates */
		__global float *beta_d,   /* dim = length x nstates */
		float sum_ab,             /* sum dot production of (alpha_d+t*nstates) and (beta_d+t*nstates) */
		int nstates,              /* number of states */
		int nsymbols,             /* number of symbols */
		int obs_t,                /* current observation */
		int t)                    /* current time */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	// unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

	if (idy < nsymbols && idx < nstates && obs_t == idy)
	{
		b_d[(idy * nstates) + idx] += alpha_d[(t * nstates) + idx] *
			beta_d[(t * nstates) + idx] / sum_ab;
	}
}
