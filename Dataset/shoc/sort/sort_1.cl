#define FPTYPE uint
#define FPVECTYPE uint4

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


// Compute a per block histogram of the occurrences of each
// digit, using a 4-bit radix (i.e. 16 possible digits).
__kernel void
reduce(__global const FPTYPE * in,
       __global FPTYPE * isums,
       const int n,
       __local FPTYPE * lmem,
       const int shift)
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

    // The per thread histogram, initially 0's.
    int digit_counts[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0 };

    // Reduce multiple elements per thread
    while (i < block_stop)
    {
        // This statement
        // 1) Loads the value in from global memory
        // 2) Shifts to the right to have the 4 bits of interest
        //    in the least significant places
        // 3) Masks any more significant bits away. This leaves us
        // with the relevant digit (which is also the index into the
        // histogram). Next increment the histogram to count this occurrence.
        digit_counts[(in[i] >> shift) & 0xFU]++;
        i += get_local_size(0);
    }

    for (int d = 0; d < 16; d++)
    {
        // Load this thread's sum into local/shared memory
        lmem[tid] = digit_counts[d];
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

        // Write result for this block to global memory
        if (tid == 0)
        {
            isums[(d * get_num_groups(0)) + get_group_id(0)] = lmem[0];
        }
    }
}
