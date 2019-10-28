/************************************************************************/
/* OpenCL Kernels for BW Algorithm in Hidden Markov Model               */
/*                                                                      */
/************************************************************************/
#define SDOT_BLOCK_SIZE        (128)
#define SDOT_BLOCK_NUM         (80)

#define MVMUL_BLOCK_SIZE       (128)
#define MVMUL_BLOCK_NUM        (64)
#define TILE_SIZE              (32)


/* mat x vec naive verion */
__kernel void mvm_non_kernel_naive(int m,
		int n,
		__global float *A,
		int lda,
		__global float *x,
		int offsetX,
		__global float *y,
		int offsetY)
{
	unsigned int i, j;
	unsigned int tid = get_local_id(0);
	unsigned int totalThreads = get_num_groups(0) * MVMUL_BLOCK_SIZE;
	unsigned int offset = MVMUL_BLOCK_SIZE * get_group_id(0);
	int n_size, m_size;

	float sum;
	if(lda == m)
	{
		n_size = n;
		m_size = m;
	} else
	{
		n_size = m;
		m_size = n;
	}

	for(i = offset + tid; i < m_size; i += totalThreads)
	{
		sum = 0.f;
		for(j = 0; j < n_size; j++)
		{
			sum += A[i * n_size + j] * x[j+offsetX];
		}
		y[i+offsetY] = sum;
	}

}
