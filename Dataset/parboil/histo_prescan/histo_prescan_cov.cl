#define OCL_NEW_BARRIER(barrierid,arg)\
{\
  atom_inc(&ocl_kernel_barrier_count[barrierid]);\
  barrier(arg);\
  if (ocl_kernel_barrier_count[barrierid]!=ocl_get_general_size()) {\
    ocl_barrier_divergence_recorder[barrierid]=1;\
  }\
  barrier(arg);\
  ocl_kernel_barrier_count[barrierid]=0;\
  barrier(arg);\
}
int ocl_get_general_size(){
  int result = 1;\
  for (int i=0; i<get_work_dim(); i++){
    result*=get_local_size(i);
  }
  return result;
}

/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#ifndef PRESCAN_THREADS
#define PRESCAN_THREADS   512
#define KB                24
#define BLOCK_X           14
#define UNROLL            16
#define BINS_PER_BLOCK    (KB * 1024)
#endif

/* Combine all the sub-histogram results into one final histogram */
__kernel void histo_prescan_kernel (__global unsigned int* input, unsigned int size, __global unsigned int* minmax, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[12];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 12; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[3];
bool private_ocl_kernel_loop_boundary_not_reached[3];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    __local float Avg[PRESCAN_THREADS];
    __local float StdDev[PRESCAN_THREADS];

    int threadIdxx = get_local_id(0);
    int blockDimx = get_local_size(0);
    int blockIdxx = get_group_id(0);
    int stride = size/(get_num_groups(0));
    int addr = blockIdxx*stride+threadIdxx;
    int end = blockIdxx*stride + stride/8; // Only sample 1/8th of the input data

    // Compute the average per thread
    float avg = 0.0;
    unsigned int count = 0;
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while (addr < end || (private_ocl_kernel_loop_boundary_not_reached[0] = false)){
private_ocl_kernel_loop_iter_counter[0]++;

      avg += input[addr];
      count++;
	  addr += blockDimx;
    }
if (private_ocl_kernel_loop_iter_counter[0] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 1);
}if (private_ocl_kernel_loop_iter_counter[0] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 2);
}if (private_ocl_kernel_loop_iter_counter[0] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[0]) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 8);
}
    avg /= count;
    Avg[threadIdxx+0] = avg;

    // Compute the standard deviation per thread
    int addr2 = blockIdxx*stride+threadIdxx;
    float stddev = 0;
    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while (addr2 < end || (private_ocl_kernel_loop_boundary_not_reached[1] = false)){
private_ocl_kernel_loop_iter_counter[1]++;

        stddev += (input[addr2]-avg)*(input[addr2]-avg);
        addr2 += blockDimx;
    }
if (private_ocl_kernel_loop_iter_counter[1] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 1);
}if (private_ocl_kernel_loop_iter_counter[1] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 2);
}if (private_ocl_kernel_loop_iter_counter[1] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[1]) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 8);
}
    stddev /= count;
    StdDev[threadIdxx+0] = sqrt(stddev);

#define SUM(stride__)\
if(threadIdxx < stride__){\
    Avg[threadIdxx] += Avg[threadIdxx+stride__];\
    StdDev[threadIdxx] += StdDev[threadIdxx+stride__];\
} \
	barrier(CLK_LOCAL_MEM_FENCE);

	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

    // Add all the averages and standard deviations from all the threads
    // and take their arithmetic average (as a simplified approximation of the
    // real average and standard deviation.
#if (PRESCAN_THREADS >= 32)    
    private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for (int stride = PRESCAN_THREADS/2; stride >= 32 || (private_ocl_kernel_loop_boundary_not_reached[2] = false); stride = stride >> 1){
private_ocl_kernel_loop_iter_counter[2]++;

	SUM(stride);
    }
if (private_ocl_kernel_loop_iter_counter[2] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 1);
}if (private_ocl_kernel_loop_iter_counter[2] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 2);
}if (private_ocl_kernel_loop_iter_counter[2] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[2]) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 8);
}
#endif
#if (PRESCAN_THREADS >= 16)
    SUM(16);
#endif
#if (PRESCAN_THREADS >= 8)
    SUM(8);
#endif
#if (PRESCAN_THREADS >= 4)
    SUM(4);
#endif
#if (PRESCAN_THREADS >= 2)
    SUM(2);
#endif

    if (threadIdxx == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

        float avg = Avg[0]+Avg[1];
	avg /= PRESCAN_THREADS;
	float stddev = StdDev[0]+StdDev[1];
	stddev /= PRESCAN_THREADS;

        // Take the maximum and minimum range from all the blocks. This will
        // be the final answer. The standard deviation is taken out to 10 sigma
        // away from the average. The value 10 was obtained empirically.
	    atom_min(minmax,((unsigned int)(avg-10*stddev))/(KB*1024));
        atom_max(minmax+1,((unsigned int)(avg+10*stddev))/(KB*1024));
    } else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}
 
for (int update_recorder_i = 0; update_recorder_i < 12; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 3; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
