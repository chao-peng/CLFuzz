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



__kernel void uniformAdd(unsigned int n, __global unsigned int *dataBase, unsigned int data_offset, __global unsigned int *interBase, unsigned int inter_offset)
{
    __local unsigned int uni;

    __global unsigned int *data = dataBase + data_offset;
    __global unsigned int *inter = interBase + inter_offset;
    unsigned int lsz0 = get_local_size(0);
    unsigned int lid0 = get_local_id(0);

    if (lid0 == 0) { uni = inter[get_group_id(0)]; }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    unsigned int g_ai = get_group_id(0)*2*lsz0 + get_local_id(0);
    unsigned int g_bi = g_ai + lsz0;

    if (g_ai < n) { data[g_ai] += uni; }
    if (g_bi < n) { data[g_bi] += uni; }
}
