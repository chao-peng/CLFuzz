typedef struct
{
	int starting;
	int no_of_edges;
}Node;



__kernel void kernel2(__global int* g_graph_mask,
		__global int* g_updating_graph_mask,
		__global int* g_graph_visited,
		__global int* g_over,
		int no_of_nodes, __global int* ocl_kernel_branch_triggered_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	unsigned int tid = get_global_id(0);
	if(tid < no_of_nodes && g_updating_graph_mask[tid] == 1)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		g_graph_mask[tid] = 1;
		g_graph_visited[tid] = 1;
		*g_over = 1;
		g_updating_graph_mask[tid] = 0;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
