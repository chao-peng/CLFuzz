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

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

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

#define UINT32_MAX (4294967295/2)
#define BITS 4
#define LNB 4

#define SORT_BS 256

//#define CONFLICT_FREE_OFFSET(index) ((index) >> LNB + (index) >> (2*LNB))
#define CONFLICT_FREE_OFFSET(index) (((unsigned int)(index) >> min((unsigned int)(LNB)+(index), (unsigned int)(32-(2*LNB))))>>(2*LNB))
#define BLOCK_P_OFFSET (4*SORT_BS+1+(4*SORT_BS+1)/16+(4*SORT_BS+1)/64)



__kernel void splitRearrange (int numElems, int iter,
                                __global unsigned int* keys_i,
                                __global unsigned int* keys_o,
                                __global unsigned int* values_i,
                                __global unsigned int* values_o,
                                __global unsigned int* histo, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder){__local int my_ocl_kernel_branch_triggered_recorder[8];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 8; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  __local unsigned int histo_s[(1<<BITS)];
  __local uint array_s[4*SORT_BS];
  const unsigned int tid = get_local_id(0);
  int index = get_group_id(0)*4*SORT_BS + 4*get_local_id(0);

  if (tid < (1<<BITS)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

    histo_s[tid] = histo[get_num_groups(0)*tid+get_group_id(0)];
  }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


  uint mine_x;
  uint mine_y;
  uint mine_z;
  uint mine_w;
  uint value_x = tid;
  uint value_y = tid;
  uint value_z = tid;
  uint value_w = tid;
  if (index < numElems){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

    mine_x = *(__global uint*)(keys_i+index);
    mine_y = *(__global uint*)(keys_i+index+1);
    mine_z = *(__global uint*)(keys_i+index+2);
    mine_w = *(__global uint*)(keys_i+index+3);
    value_x = *(__global uint*)(values_i+index);
    value_y = *(__global uint*)(values_i+index+1);
    value_z = *(__global uint*)(values_i+index+2);
    value_w = *(__global uint*)(values_i+index+3);
  } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

    mine_x = UINT32_MAX;
    mine_y = UINT32_MAX;
    mine_z = UINT32_MAX;
    mine_w = UINT32_MAX;
  }

  uint masks_x = (mine_x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
  uint masks_y = (mine_y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
  uint masks_z = (mine_z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
  uint masks_w = (mine_w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);

  ((__local uint*) array_s)[4*get_local_id(0)  ] = masks_x;
  ((__local uint*) array_s)[4*get_local_id(0)+1] = masks_y;
  ((__local uint*) array_s)[4*get_local_id(0)+2] = masks_z;
  ((__local uint*) array_s)[4*get_local_id(0)+3] = masks_w;

  OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE); //__syncthreads();

  uint new_index_x = histo_s[masks_x];
  uint new_index_y = histo_s[masks_y];
  uint new_index_z = histo_s[masks_z];
  uint new_index_w = histo_s[masks_w];

  int i = 4*get_local_id(0)-1;

  private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while (i >= 0 || (private_ocl_kernel_loop_boundary_not_reached[0] = false)){
private_ocl_kernel_loop_iter_counter[0]++;

    if (array_s[i] == masks_x){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

      new_index_x = new_index_x+1;
      i--;
    } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

      break;
    }
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

  new_index_y = (masks_y == masks_x) ? new_index_x+1 : new_index_y;
  new_index_z = (masks_z == masks_y) ? new_index_y+1 : new_index_z;
  new_index_w = (masks_w == masks_z) ? new_index_z+1 : new_index_w;

  if (index < numElems){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

    keys_o[new_index_x] = mine_x;
    values_o[new_index_x] = value_x;

    keys_o[new_index_y] = mine_y;
    values_o[new_index_y] = value_y;

    keys_o[new_index_z] = mine_z;
    values_o[new_index_z] = value_z;

    keys_o[new_index_w] = mine_w;
    values_o[new_index_w] = value_w;
  }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 8; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
