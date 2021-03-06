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



__kernel void uniformAdd(unsigned int n, __global unsigned int *dataBase, unsigned int data_offset, __global unsigned int *interBase, unsigned int inter_offset, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[6];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 6; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __local unsigned int uni;

    __global unsigned int *data = dataBase + data_offset;
    __global unsigned int *inter = interBase + inter_offset;
    unsigned int lsz0 = get_local_size(0);
    unsigned int lid0 = get_local_id(0);

    if (lid0 == 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
 uni = inter[get_group_id(0)]; }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE); //__syncthreads();

    unsigned int g_ai = get_group_id(0)*2*lsz0 + get_local_id(0);
    unsigned int g_bi = g_ai + lsz0;

    if (g_ai < n) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
 data[g_ai] += uni; }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
    if (g_bi < n) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
 data[g_bi] += uni; }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
