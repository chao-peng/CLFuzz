/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)


/* Re-estimate B values */
__kernel void est_b_dev( __global float *b_d,         /* dim = nsymbols x nstates */
		__global float *gamma_sum_d, /* lth = nstates */
		int nstates,                 /* number of states */
		int nsymbols)                /* number of symbols */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	// unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

	if (idy < nsymbols && idx < nstates)
	{
		b_d[(idy * nstates) + idx] = b_d[(idy * nstates) + idx] /
			gamma_sum_d[idx];
	}
}
