//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

__kernel void Fan1(__global float *m_dev,
                  __global float *a_dev,
                  __global float *b_dev,
                  const int size,
                  const int t, __global int* ocl_kernel_branch_triggered_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int globalId = get_global_id(0);

    if (globalId < size-1-t) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

         *(m_dev + size * (globalId + t + 1)+t) = *(a_dev + size * (globalId + t + 1) + t) / *(a_dev + size * t + t);
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
