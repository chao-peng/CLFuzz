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

// This kernel scans the contents of local memory using a work
// inefficient, but highly parallel Kogge-Stone style scan.
// Set exclusive to 1 for an exclusive scan or 0 for an inclusive scan
inline FPTYPE scanLocalMem(FPTYPE val, __local FPTYPE* lmem, int exclusive, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count, __local int* my_ocl_kernel_loop_recorder)
{
  int private_ocl_kernel_loop_iter_counter[1];
  bool private_ocl_kernel_loop_boundary_not_reached[1];
    // Set first half of local memory to zero to make room for scanning
    int idx = get_local_id(0);
    lmem[idx] = 0.0f;

    // Set second half to block sums from global memory, but don't go out
    // of bounds
    idx += get_local_size(0);
    lmem[idx] = val;
    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

    // Now, perform Kogge-Stone scan
    FPTYPE t;
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int i = 1; i < get_local_size(0) || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i *= 2)
    {
private_ocl_kernel_loop_iter_counter[0]++;

        t = lmem[idx -  i]; OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
        lmem[idx] += t;     OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);
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
    return lmem[idx-exclusive];
}

__kernel void
top_scan(__global FPTYPE * isums, const int n, __local FPTYPE * lmem, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    FPTYPE val = get_local_id(0) < n ? isums[get_local_id(0)] : 0.0f;
    val = scanLocalMem(val, lmem, 1, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);

    if (get_local_id(0) < n)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        isums[get_local_id(0)] = val;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) {
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]);
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) {
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]);
}
}
