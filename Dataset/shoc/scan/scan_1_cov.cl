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

#ifdef SINGLE_PRECISION
#define FPTYPE float
#define FPVECTYPE float4
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#define FPVECTYPE double4
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#define FPVECTYPE double4
#endif

#define FPTYPE float
#define FPVECTYPE float4

__kernel void
reduce(__global const FPTYPE * in,
       __global FPTYPE * isums,
       const int n,
       __local FPTYPE * lmem, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
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

    // First, calculate the bounds of the region of the array
    // that this block will sum.  We need these regions to match
    // perfectly with those in the bottom-level scan, so we index
    // as if vector types of length 4 were in use.  This prevents
    // errors due to slightly misaligned regions.
    int region_size = ((n / 4) / get_num_groups(0)) * 4;
    int block_start = get_group_id(0) * region_size;

    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ?
        n : block_start + region_size;

    // Calculate starting index for this thread/work item
    int tid = get_local_id(0);
    int i = block_start + tid;

    FPTYPE sum = 0.0f;

    // Reduce multiple elements per thread
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while (i < block_stop || (private_ocl_kernel_loop_boundary_not_reached[0] = false))
    {
private_ocl_kernel_loop_iter_counter[0]++;

        sum += in[i];
        i += get_local_size(0);
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
    // Load this thread's sum into local/shared memory
    lmem[tid] = sum;
    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

    // Reduce the contents of shared/local memory
    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (unsigned int s = get_local_size(0) / 2; s > 0 || (private_ocl_kernel_loop_boundary_not_reached[1] = false); s >>= 1)
    {
private_ocl_kernel_loop_iter_counter[1]++;

        if (tid < s)
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

            lmem[tid] += lmem[tid + s];
        }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
        OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
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

    OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);
    // Write result for this block to global memory
    if (tid == 0)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

        isums[get_group_id(0)] = lmem[0];
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
