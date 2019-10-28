/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* Calculate alpha variables */
__kernel void calc_alpha_dev( int nstates,             /* number of states */
		__global float *alpha_d, /* dim = length x nstates */
		int offset,              /* offset for alpha_d */
		__global float *b_d,     /* dim = nsymbols x nstates */
		int obs_t)               /* current observation */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nstates)
	{
		alpha_d[offset + idx] = alpha_d[offset + idx] * b_d[(obs_t * nstates) + idx];
	}
}
