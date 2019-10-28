//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;



__kernel void Fan2(__global float *m_dev,
                  __global float *a_dev,
                  __global float *b_dev,
                  const int size,
                  const int t, __global int* ocl_kernel_branch_triggered_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	 int globalId = get_global_id(0);

	 int globalIdx = get_global_id(0);
	 int globalIdy = get_global_id(1);
      if (globalIdx < size-1-t && globalIdy < size-t) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

         a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];

 	    if(globalIdy == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

 		   b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
 	    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

 	 }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

//   One dimensional
// 	 int globalIdx = globalId % size;
// 	 int globalIdy = globalId / size;
//
// 	 if (globalIdx < size-1-t && globalIdy < size-t) {
//          a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];
// 	 }
// 	 if(globalIdy == 0){
//  		   b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
//      }

for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
