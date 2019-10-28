#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define get_queue_index(tid) ((tid%NUM_P_PER_MP))
#define get_queue_offset(tid) ((tid%NUM_P_PER_MP)*W_Q_SIZE)

//S. Xiao and W. Feng, .Inter-block GPU communication via fast barrier
//synchronization,.Technical Report TR-09-19,
//Dept. of Computer Science, Virginia Tech
// ****************************************************************************
// Function: __gpu_sync
//
// Purpose:
//   Implements global barrier synchronization across thread blocks. Thread
//   blocks must be limited to number of multiprocessors available
//
// Arguments:
//   blocks_to_synch: the number of blocks across which to implement the barrier
//   g_mutex: keeps track of number of blocks that are at barrier
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************



//An Effective GPU Implementation of Breadth-First Search, Lijuan Luo,
//Martin Wong,Wen-mei Hwu ,
//Department of Electrical and Computer Engineering,
//University of Illinois at Urbana-Champaign
// ****************************************************************************
// Function: BFS_kernel_one_block
//
// Purpose:
//   Perform BFS on the given graph when the frontier length is within one
//   thread block (i.e max number of threads per block)
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier_len: length of the given frontier array
//   visited: mask that tells if a vertex is currently in frontier
//   cost: array that stores the cost to visit each vertex
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//   b_q: block level queue
//   b_q2: alterante block level queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************


// ****************************************************************************
// Function: Frontier_copy
//
// Purpose:
//   Copy frontier2 data to frontier
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier2: alternate frontier array
//   frontier_length: length of the frontier array
//   g_mutex: mutex for implementing global barrier
//   g_mutex2: gives the starting value of the g_mutex used in global barrier
//   g_q_offsets: gives the offset of a block in the global queue
//   g_q_size: size of the global queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void Frontier_copy(
    __global unsigned int *frontier,
    __global unsigned int *frontier2,
    __global unsigned int *frontier_length,
    __global volatile int *g_mutex,
    __global volatile int *g_mutex2,
    __global volatile int *g_q_offsets,
    __global volatile int *g_q_size)
{
    unsigned int tid=get_global_id(0);

    if(tid<*frontier_length)
    {
        frontier[tid]=frontier2[tid];
    }
}
