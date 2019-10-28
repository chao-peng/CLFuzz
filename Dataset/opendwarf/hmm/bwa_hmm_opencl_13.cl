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
		int nsymbols)        /* number of symbols */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	// unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

	if (idx < nstates && idy < nsymbols)
	{
		if (b_d[(idy * nstates) + idx] == 0)
		{
			b_d[(idy * nstates) + idx] = 1e-10;
		}
		else
		{
			b_d[(idy * nstates) + idx] = b_d[(idy * nstates) + idx] / c_d[idx];
		}
	}
}
