#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable

#define __THREAD_FENCE_USED__

typedef struct {
	int nposi, nposj;
	int nmaxpos;
	float fmaxscore;
	int noutputlen;
}   MAX_INFO;

#define PATH_END 0
#define COALESCED_OFFSET 32


__kernel void setZero(__global char *a,
		int arraySize)
{
	unsigned int index = get_global_id(0);
	if (index < arraySize)
	{
		a[index] = 0;
	}
}
