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

__kernel void splitSort(int numElems, int iter,
                                 __global unsigned int* keys,
                                 __global unsigned int* values,
                                 __global unsigned int* histo, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[14];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 14; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[7];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 7; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[3];
bool private_ocl_kernel_loop_boundary_not_reached[3];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __local unsigned int flags[BLOCK_P_OFFSET];
    __local unsigned int histo_s[1<<BITS];

    const unsigned int tid = get_local_id(0);
    const unsigned int gid = get_group_id(0)*4*SORT_BS+4*get_local_id(0);

    // Copy input to shared mem. Assumes input is always even numbered
    uint lkey_x = UINT32_MAX;
    uint lkey_y = UINT32_MAX;
    uint lkey_z = UINT32_MAX;
    uint lkey_w = UINT32_MAX;
    uint lvalue_x = tid;
    uint lvalue_y = tid;
    uint lvalue_z = tid;
    uint lvalue_w = tid;
    if (gid < numElems){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

      lkey_x = *(keys+gid);
      lkey_y = *(keys+gid+1);
      lkey_z = *(keys+gid+2);
      lkey_w = *(keys+gid+3);
      lvalue_x = *(values+gid);
      lvalue_y = *(values+gid+1);
      lvalue_z = *(values+gid+2);
      lvalue_w = *(values+gid+3);
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

    if(tid < (1<<BITS)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

      histo_s[tid] = 0;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE); //__syncthreads();

    atom_add(histo_s+((lkey_x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey_y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey_z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey_w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);

    int _x = 4*tid;
    int _y = 4*tid+1;
    int _z = 4*tid+2;
    int _w = 4*tid+3;

    // promoted due to CEAN bug
    unsigned int _i = tid;
    unsigned int ai = tid;
    unsigned int bi = tid;
    unsigned int t = tid;
    unsigned int last = tid;

    unsigned int lsz0 = get_local_size(0);
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int i=BITS*iter; i<BITS*(iter+1) || (private_ocl_kernel_loop_boundary_not_reached[0] = false);i++){
private_ocl_kernel_loop_iter_counter[0]++;

      const uint flag_x = (lkey_x>>i)&0x1;
      const uint flag_y = (lkey_y>>i)&0x1;
      const uint flag_z = (lkey_z>>i)&0x1;
      const uint flag_w = (lkey_w>>i)&0x1;

      flags[_x+CONFLICT_FREE_OFFSET(_x)] = 1<<(16*flag_x);
      flags[_y+CONFLICT_FREE_OFFSET(_y)] = 1<<(16*flag_y);
      flags[_z+CONFLICT_FREE_OFFSET(_z)] = 1<<(16*flag_z);
      flags[_w+CONFLICT_FREE_OFFSET(_w)] = 1<<(16*flag_w);

      // scan (flags);
      {
        __local unsigned int* s_data = flags;
        unsigned int thid = tid;

        OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE); //__syncthreads();

        s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
        s_data[2*(lsz0+thid)+1+CONFLICT_FREE_OFFSET(2*(lsz0+thid)+1)] += s_data[2*(lsz0+thid)+CONFLICT_FREE_OFFSET(2*(lsz0+thid))];

        unsigned int stride = 2;
        private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (unsigned int d = lsz0; d > 0 || (private_ocl_kernel_loop_boundary_not_reached[1] = false); d >>= 1)
        {
private_ocl_kernel_loop_iter_counter[1]++;

          OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE); //__syncthreads();

          if (thid < d)
          {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

            _i  = 2*stride*thid;
            ai = _i + stride - 1;
            bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
          }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

          stride *= 2;
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

        if (thid == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

#if 0
          last = 4*lsz0-1;
          last += CONFLICT_FREE_OFFSET(last);
          s_data[4*lsz0+CONFLICT_FREE_OFFSET(4*lsz0)] = s_data[last];
          s_data[last] = 0;
#else
          #define LAST  ((4*lsz0-1) + CONFLICT_FREE_OFFSET((4*lsz0-1)))
          s_data[(4*lsz0+CONFLICT_FREE_OFFSET(4*lsz0))] = s_data[LAST];
          s_data[LAST] = 0;
#endif
        }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}

        private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for (unsigned int d = 1; d <= lsz0 || (private_ocl_kernel_loop_boundary_not_reached[2] = false); d *= 2)
        {
private_ocl_kernel_loop_iter_counter[2]++;

          stride >>= 1;

          OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE); //__syncthreads();

          if (thid < d)
          {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

            _i  = 2*stride*thid;
            ai = _i + stride - 1;
            bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
          }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
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
        OCL_NEW_BARRIER(4,CLK_LOCAL_MEM_FENCE); //__syncthreads();

        unsigned int temp = s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
        s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)] = s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)];
        s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += temp;

        unsigned int temp2 = s_data[2*(lsz0+thid)+CONFLICT_FREE_OFFSET(2*(lsz0+thid))];
        s_data[2*(lsz0+thid)+CONFLICT_FREE_OFFSET(2*(lsz0+thid))] = s_data[2*(lsz0+thid)+1+CONFLICT_FREE_OFFSET(2*(lsz0+thid)+1)];
        s_data[2*(lsz0+thid)+1+CONFLICT_FREE_OFFSET(2*(lsz0+thid)+1)] += temp2;

        OCL_NEW_BARRIER(5,CLK_LOCAL_MEM_FENCE); //__syncthreads();
      }

      _x = (flags[_x+CONFLICT_FREE_OFFSET(_x)]>>(16*flag_x))&0xFFFF;
      _y = (flags[_y+CONFLICT_FREE_OFFSET(_y)]>>(16*flag_y))&0xFFFF;
      _z = (flags[_z+CONFLICT_FREE_OFFSET(_z)]>>(16*flag_z))&0xFFFF;
      _w = (flags[_w+CONFLICT_FREE_OFFSET(_w)]>>(16*flag_w))&0xFFFF;

      unsigned short offset = flags[4*lsz0+CONFLICT_FREE_OFFSET(4*lsz0)]&0xFFFF;
      _x += (flag_x) ? offset : 0;
      _y += (flag_y) ? offset : 0;
      _z += (flag_z) ? offset : 0;
      _w += (flag_w) ? offset : 0;

      OCL_NEW_BARRIER(6,CLK_LOCAL_MEM_FENCE); //__syncthreads();
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

    // Write result.
    if (gid < numElems){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

      keys[get_group_id(0)*4*SORT_BS+_x] = lkey_x;
      keys[get_group_id(0)*4*SORT_BS+_y] = lkey_y;
      keys[get_group_id(0)*4*SORT_BS+_z] = lkey_z;
      keys[get_group_id(0)*4*SORT_BS+_w] = lkey_w;

      values[get_group_id(0)*4*SORT_BS+_x] = lvalue_x;
      values[get_group_id(0)*4*SORT_BS+_y] = lvalue_y;
      values[get_group_id(0)*4*SORT_BS+_z] = lvalue_z;
      values[get_group_id(0)*4*SORT_BS+_w] = lvalue_w;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}
    if (tid < (1<<BITS)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

      histo[get_num_groups(0)*tid+get_group_id(0)] = histo_s[tid];
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 14; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 3; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
