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
inline FPTYPE scanLocalMem(FPTYPE val, __local FPTYPE* lmem, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int idx = get_local_id(0);
    lmem[idx] = 0.0f;

    // Set second half to block sums from global memory, but don't go out
    // of bounds
    idx += get_local_size(0);
    lmem[idx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Now, perform Kogge-Stone scan
    FPTYPE t;
    for (int i = 1; i < get_local_size(0); i *= 2)
    {
        t = lmem[idx -  i]; barrier(CLK_LOCAL_MEM_FENCE);
        lmem[idx] += t;     barrier(CLK_LOCAL_MEM_FENCE);
    }
    return lmem[idx-exclusive];
}

__kernel void
top_scan(__global FPTYPE * isums, const int n, __local FPTYPE * lmem)
{
    FPTYPE val = get_local_id(0) < n ? isums[get_local_id(0)] : 0.0f;
    val = scanLocalMem(val, lmem, 1);

    if (get_local_id(0) < n)
    {
        isums[get_local_id(0)] = val;
    }
}
