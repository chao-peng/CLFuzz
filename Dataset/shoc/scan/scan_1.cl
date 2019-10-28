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
       __local FPTYPE * lmem)
{
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
    while (i < block_stop)
    {
        sum += in[i];
        i += get_local_size(0);
    }
    // Load this thread's sum into local/shared memory
    lmem[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce the contents of shared/local memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            lmem[tid] += lmem[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Write result for this block to global memory
    if (tid == 0)
    {
        isums[get_group_id(0)] = lmem[0];
    }
}
