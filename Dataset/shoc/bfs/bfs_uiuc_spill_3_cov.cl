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
__kernel void BFS_kernel_multi_block(

    volatile __global unsigned int *frontier,
    unsigned int frontier_len,
    volatile __global unsigned int *frontier2,
    volatile __global int *visited,
    volatile __global unsigned int *cost,
    __global unsigned int *edgeArray,
    __global unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile __global unsigned int *frontier_length,
    const unsigned int max_local_mem,

    volatile __local unsigned int *b_q, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[16];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 16; ++ocl_kernel_init_i) {
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

    volatile __local unsigned int b_q_length[1];
    volatile __local unsigned int b_offset[1];

    //get the threadId
    unsigned int tid=get_global_id(0);
    unsigned int lid=get_local_id(0);

    //initialize the block queue length
    if (lid == 0)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        b_q_length[0]=0;
        b_offset[0]=0;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if(tid<frontier_len)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

        //get the nodes to traverse from block queue
        unsigned int node_to_process=frontier[tid];
        visited[node_to_process]=0;
        //get the offsets of the vertex in the edge list
        unsigned int offset=edgeArray[node_to_process];
        unsigned int next=edgeArray[node_to_process+1];

        //Iterate through the neighbors of the vertex
        private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while(offset<next || (private_ocl_kernel_loop_boundary_not_reached[0] = false))
        {
private_ocl_kernel_loop_iter_counter[0]++;

            //get neighbor
            unsigned int nid=edgeArrayAux[offset];
            //get its cost
            unsigned int v=atomic_min(&cost[nid],cost[node_to_process]+1);
            //if cost is less than previously set add to frontier
            if(v>cost[node_to_process]+1)
            {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

                int is_in_frontier=atomic_xchg(&visited[nid],1);
                //if node already in frontier do nothing
                if(is_in_frontier==0)
                {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

                        //increment the warp queue size
                        unsigned int t=atomic_add(&b_q_length[0],1);
                        if(t<max_local_mem)
                        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

                            b_q[t]=nid;
                        }
                        //write to global memory if shared memory full
                        else
                        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);

                            int off=atomic_add(frontier_length,1);
                            frontier2[off]=nid;
                        }
                }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
            }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
            offset++;
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

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
    OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    //get block queue offset in global queue
    if(lid==0)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

        if(b_q_length[0] > max_local_mem)
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

                b_q_length[0]=max_local_mem;
        }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}
        b_offset[0]=atomic_add(frontier_length,b_q_length[0]);
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}

    OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    //copy block queue to global queue
    if(lid < b_q_length[0])
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);
frontier2[lid+b_offset[0]]=b_q[lid];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}


for (int update_recorder_i = 0; update_recorder_i < 16; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
