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
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif
#define FPTYPE float
__kernel void
reduce(__global const FPTYPE *g_idata, __global FPTYPE *g_odata,
       __local FPTYPE* sdata, const unsigned int n, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const unsigned int tid = get_local_id(0);
    unsigned int i = (get_group_id(0)*(get_local_size(0)*2)) + tid;
    const unsigned int gridSize = get_local_size(0)*2*get_num_groups(0);
    const unsigned int blockSize = get_local_size(0);

    sdata[tid] = 0;

    // Reduce multiple elements per thread, strided by grid size
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while (i < n || (private_ocl_kernel_loop_boundary_not_reached[0] = false))
    {
private_ocl_kernel_loop_iter_counter[0]++;

        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
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
    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (unsigned int s = blockSize / 2; s > 0 || (private_ocl_kernel_loop_boundary_not_reached[1] = false); s >>= 1)
    {
private_ocl_kernel_loop_iter_counter[1]++;

        if (tid < s)
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

            sdata[tid] += sdata[tid + s];
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

    // Write result back to global memory
    if (tid == 0)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

        g_odata[get_group_id(0)] = sdata[0];
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


// Currently, CPUs on Snow Leopard only support a work group size of 1
// So, we have a separate version of the kernel which doesn't use
// local memory. This version is only used when the maximum
// supported local group size is 1.
