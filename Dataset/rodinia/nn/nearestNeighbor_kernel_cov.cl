//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

__kernel void NearestNeighbor(__global LatLong *d_locations,
							  __global float *d_distances,
							  const int numRecords,
							  const float lat,
							  const float lng, __global int* ocl_kernel_branch_triggered_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	 int globalId = get_global_id(0);
							  
     if (globalId < numRecords) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

         __global LatLong *latLong = d_locations+globalId;
    
         __global float *dist=d_distances+globalId;
         *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	 }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
