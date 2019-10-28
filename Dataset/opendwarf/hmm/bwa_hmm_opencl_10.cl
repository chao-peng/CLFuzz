/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)


/* Normalize A matrix */
__kernel void scale_a_dev(  __global  float *a_d, /* dim = nstates x nstates */
		__global  float *c_d, /* lth = nstates, matrix-vector product of a_d and ones_n_d */
		int nstates)          /* number of states */
{
	// unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	// unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

	if (idx < nstates && idy < nstates)
	{
		a_d[(idy * nstates) + idx] = a_d[(idy * nstates) + idx] / c_d[idy];
	}
}
