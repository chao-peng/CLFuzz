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
		int t)                  /* current time */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nstates)
	{
		beta_d[(t * nstates) + idx] = beta_d[((t + 1) * nstates) + idx] *
			b_d[(obs_t * nstates) + idx] / scale_t;
	}
}
