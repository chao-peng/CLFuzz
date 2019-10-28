/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)



/* dot production naive version */
__kernel void s_dot_kernel_naive(int n,                        /* number of elements */
		__global float *paramA,       /* first vector A */
		int offsetA,                  /* offset of vector A */
		__global float *paramB,       /* second vector B */
		int offsetB,                  /* offset of vector B */
		__global float *partialSum_d) /* memory space for partial sum */
{
	unsigned int i;
	unsigned int tid = get_local_id(0);
	unsigned int totalThreads = get_num_groups(0) * SDOT_BLOCK_SIZE;
	unsigned int offset = SDOT_BLOCK_SIZE * get_group_id(0);

	for(i = offset + tid; i < n; i += totalThreads)
	{
		partialSum_d[i] = paramA[offsetA + i] * paramB[offsetB + i];
	}
}
