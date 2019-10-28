/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)

/* Initialize ones vector */
__kernel void init_ones_dev(  __global float *ones_s_d,  /* lth = nsymbols */
		int nsymbols)              /* number of symbols */
{
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

	if (idx < nsymbols)
	{
		ones_s_d[idx] = 1.0f;
	}
}
