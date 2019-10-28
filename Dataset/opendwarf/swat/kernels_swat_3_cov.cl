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
		int arraySize, __global int* ocl_kernel_branch_triggered_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	unsigned int index = get_global_id(0);
	if (index < arraySize)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		a[index] = 0;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
