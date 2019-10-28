/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

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

#define BLOCK_SIZE 1024
#define GRID_SIZE 65535
#define NUM_BANKS 16

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////


__kernel void scan_inter1_kernel(__global unsigned int* data, unsigned int iter)
{
    __local unsigned int s_data[DYN_LOCAL_MEM_SIZE];

    unsigned int thid = get_local_id(0);
    unsigned int gthid = get_global_id(0);
    unsigned int gi = 2*iter*gthid;
    unsigned int g_ai = gi + iter - 1;
    unsigned int g_bi = g_ai + iter;

    unsigned int s_ai = 2*thid;
    unsigned int s_bi = 2*thid + 1;
    s_data[s_ai] = data[g_ai];
    s_data[s_bi] = data[g_bi];

    // promoted due to CEAN bug
    unsigned int i = thid;
    unsigned int ai = thid;
    unsigned int bi = thid;

    unsigned int stride = 1;
    unsigned int lsz0 = get_local_size(0);
    for (unsigned int d = lsz0; d > 0; d >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

      if (thid < d) {
        i  = 2*stride*thid;
        ai = i + stride - 1;
        bi = ai + stride;
        s_data[bi] += s_data[ai];
      }

      stride *= 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    data[g_ai] = s_data[s_ai];
    data[g_bi] = s_data[s_bi];
}
