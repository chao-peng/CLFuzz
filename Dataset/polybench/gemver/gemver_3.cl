/**
 * gemver.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;



__kernel void gemver_kernel3(__global DATA_TYPE *A, __global DATA_TYPE *X, __global DATA_TYPE *w, DATA_TYPE alpha, int n)
{
	int i = get_global_id(0);

	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++)
		{
			w[i] += alpha * A[i*n + j] * X[j];
		}
	}
}
