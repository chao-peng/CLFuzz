/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* Initialize alpha variables */
__kernel void init_alpha_dev( __global float *b_d,      /* dim = nsymbols x nstates */
		__global  float *pi_d,    /* lth = nstates */
		int nstates,              /* number of states */
		__global float *alpha_d,  /* dim = length x nstates */
		__global float *ones_n_d, /* lth = nstates */
		int obs_t)                /* current observation */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nstates)
	{
		alpha_d[idx] = pi_d[idx] * b_d[(obs_t * nstates) + idx];
		ones_n_d[idx] = 1.0f;
	}
}
