typedef struct
{
	int starting;
	int no_of_edges;
}Node;



__kernel void kernel2(__global int* g_graph_mask,
		__global int* g_updating_graph_mask,
		__global int* g_graph_visited,
		__global int* g_over,
		int no_of_nodes)
{
	unsigned int tid = get_global_id(0);
	if(tid < no_of_nodes && g_updating_graph_mask[tid] == 1)
	{
		g_graph_mask[tid] = 1;
		g_graph_visited[tid] = 1;
		*g_over = 1;
		g_updating_graph_mask[tid] = 0;
	}
}
