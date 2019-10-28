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

#ifndef CUTOFF2_VAL
#define CUTOFF2_VAL 6.250000
#define CUTOFF_VAL 2.500000
#define CEIL_CUTOFF_VAL 3.000000
#define GRIDSIZE_VAL1 256
#define GRIDSIZE_VAL2 256
#define GRIDSIZE_VAL3 256
#define SIZE_XY_VAL 65536
#define ONE_OVER_CUTOFF2_VAL 0.160000
#endif

#ifndef DYN_LOCAL_MEM_SIZE
#define DYN_LOCAL_MEM_SIZE 1092
#endif

#define BLOCK_SIZE 1024
#define GRID_SIZE 65535
#define NUM_BANKS 16

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
__kernel void scan_L1_kernel(unsigned int n, __global unsigned int* dataBase, unsigned int data_offset, __global unsigned int* interBase, unsigned int inter_offset, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[10];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 10; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __local unsigned int s_data[BLOCK_SIZE];

    __global unsigned int *data = dataBase + data_offset;
    __global unsigned int *inter = interBase + inter_offset;

    unsigned int lsz0;

    lsz0 = get_local_size(0);

    unsigned int thid = get_local_id(0);
    unsigned int g_ai = get_group_id(0)*2*lsz0 + get_local_id(0);
    unsigned int g_bi = g_ai + lsz0;

    unsigned int s_ai = thid;
    unsigned int s_bi = thid + lsz0;

    s_data[s_ai+0] = (g_ai < n) ? data[g_ai+0] : 0;
    s_data[s_bi+0] = (g_bi < n) ? data[g_bi+0] : 0;

    unsigned int stride = 1;

    // promoted due to CEAN bug
    unsigned int i = thid;
    unsigned int ai = thid;
    unsigned int bi = thid;
    unsigned int t = thid;

    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (unsigned int d = lsz0; d > 0 || (private_ocl_kernel_loop_boundary_not_reached[0] = false); d >>= 1) {
private_ocl_kernel_loop_iter_counter[0]++;


      OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE); //__syncthreads();

      if (thid < d) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        i  = 2*stride*thid;
        ai = i + stride - 1;
        bi = ai + stride;
        s_data[bi] += s_data[ai];
      }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
      stride *= 2;
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

    int gid0 = get_group_id(0);
    if (thid == 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

      // unsigned int last;
      // last = lsz0*2 -1;
      inter[gid0] = s_data[(lsz0*2-1)];
      s_data[(lsz0*2-1)] = 0;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (unsigned int d = 1; d <= lsz0 || (private_ocl_kernel_loop_boundary_not_reached[1] = false); d *= 2) {
private_ocl_kernel_loop_iter_counter[1]++;

      stride >>= 1;

      OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE); //__syncthreads();

      if (thid < d) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

        i  = 2*stride*thid;
        ai = i + stride - 1;
        bi = ai + stride;
        t  = s_data[ai];
        s_data[ai] = s_data[bi];
        s_data[bi] += t;
      }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
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

    OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE); //__syncthreads();

    if (g_ai < n) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);
 data[g_ai] = s_data[s_ai]; }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
    if (g_bi < n) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);
 data[g_bi] = s_data[s_bi]; }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 10; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
