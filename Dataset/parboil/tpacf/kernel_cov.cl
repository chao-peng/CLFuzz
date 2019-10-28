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
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include "model.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

  __kernel
void gen_hists(__global hist_t* histograms, __global float* all_x_data,
    __constant float* dev_binb, int NUM_SETS, int NUM_ELEMENTS, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
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

  __global float* all_y_data = all_x_data + NUM_ELEMENTS*(NUM_SETS+1);
  __global float* all_z_data = all_y_data + NUM_ELEMENTS*(NUM_SETS+1);

  unsigned int bx = get_group_id(0);
  unsigned int tid = get_local_id(0);
  bool do_self = (bx < (NUM_SETS + 1));

  __global hist_t* block_histogram = histograms + NUM_BINS * bx;
  __global float* data_x;
  __global float* data_y;
  __global float* data_z;
  __global float* random_x;
  __global float* random_y;
  __global float* random_z;

  float distance;
  float random_x_s;
  float random_y_s;
  float random_z_s;

  unsigned int bin_index;
  // XXX: HSK: Bad trick to walkaround the compiler bug
  unsigned int min = get_local_id(0); // 0
  unsigned int max = get_local_id(0); // NUM_BINS

  if(tid < NUM_BINS)
  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

    block_histogram[tid] = 0;
  }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
  OCL_NEW_BARRIER(0,CLK_GLOBAL_MEM_FENCE);

  // Get pointers set up
  if( !do_self)
  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

    data_x = all_x_data;
    data_y = all_y_data;
    data_z = all_z_data;

    random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
    random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
    random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
  }
  else
  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

    random_x = all_x_data + NUM_ELEMENTS * (bx);
    random_y = all_y_data + NUM_ELEMENTS * (bx);
    random_z = all_z_data + NUM_ELEMENTS * (bx);

    data_x = random_x;
    data_y = random_y;
    data_z = random_z;
  }

  // Iterate over all random points
  private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(unsigned int j = 0; j < NUM_ELEMENTS || (private_ocl_kernel_loop_boundary_not_reached[0] = false); j += BLOCK_SIZE)
  {
private_ocl_kernel_loop_iter_counter[0]++;

    if(tid + j < NUM_ELEMENTS)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

      random_x_s = random_x[tid + j];
      random_y_s = random_y[tid + j];
      random_z_s = random_z[tid + j];
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

    // Iterate over all data points
    // If do_self, then use a tighter bound on the number of data points.
    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for(unsigned int k = 0;
        k < NUM_ELEMENTS && (do_self ? k < j + BLOCK_SIZE : 1) || (private_ocl_kernel_loop_boundary_not_reached[1] = false); k++)
    {
private_ocl_kernel_loop_iter_counter[1]++;

      // do actual calculations on the values:
      distance = data_x[k] * random_x_s +
        data_y[k] * random_y_s +
        data_z[k] * random_z_s ;

      // run binary search to find bin_index
#if 0 /* XXX: HSK: Bad trick to walkaround the compiler bug */
      min = 0;
      max = NUM_BINS;
#else
      if (get_local_id(0) >= 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

        min = 0;
        max = NUM_BINS;
      }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
#endif
      {
        unsigned int k2;

        private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
while (max > min+1 || (private_ocl_kernel_loop_boundary_not_reached[2] = false))
        {
private_ocl_kernel_loop_iter_counter[2]++;

          k2 = (min + max) / 2;
          // k2 = (min + max) >> 1;
          if (distance >= dev_binb[k2])
            {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);
max = k2;
}
          else
            {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
min = k2;
}
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
        bin_index = max - 1;
      }

      if((distance < dev_binb[min]) && (distance >= dev_binb[max]) &&
          (!do_self || (tid + j > k)) && ((tid + j) < NUM_ELEMENTS))
      {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

        atom_inc(&(block_histogram[bin_index]));
      }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
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
for (int update_recorder_i = 0; update_recorder_i < 12; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 3; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

// **===-----------------------------------------------------------===**

#endif // _PRESCAN_CU_
