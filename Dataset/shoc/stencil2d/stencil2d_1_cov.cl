
// define types based on compiler "command line"
#if defined(SINGLE_PRECISION)
#define VALTYPE float
#elif defined(K_DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VALTYPE double
#elif defined(AMD_DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VALTYPE double

#endif
#define VALTYPE float

inline
int
ToGlobalRow( int gidRow, int lszRow, int lidRow , __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global row (without halo)
    return gidRow*lszRow + lidRow;
}

inline
int
ToGlobalCol( int gidCol, int lszCol, int lidCol , __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global column (without halo)
    return gidCol*lszCol + lidCol;
}


inline
int
ToFlatHaloedIdx( int row, int col, int rowPitch , __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
    // assumes input coordinates and dimensions are logical (without halo)
    // and a halo of width 1
    return (row + 1)*(rowPitch + 2) + (col + 1);
}


inline
int
ToFlatIdx( int row, int col, int pitch , __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
    return row * pitch + col;
}


__kernel
void
CopyRect( __global VALTYPE* dest,
            int doffset,
            int dpitch,
            __global VALTYPE* src,
            int soffset,
            int spitch,
            int width,
            int height , __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int gsz = get_global_size(0);
    int lsz = get_local_size(0);
    int grow = gid * lsz + lid;

    if( grow < height )
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for( int c = 0; c < width || (private_ocl_kernel_loop_boundary_not_reached[0] = false); c++ )
        {
private_ocl_kernel_loop_iter_counter[0]++;

            (dest + doffset)[ToFlatIdx(grow,c,dpitch, my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder)] = (src + soffset)[ToFlatIdx(grow,c,spitch, my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder)];
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
