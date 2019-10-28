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

#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__kernel void dynproc_kernel (int iteration,
                              __global int* gpuWall,
                              __global int* gpuSrc,
                              __global int* gpuResults,
                              int cols,
                              int rows,
                              int startStep,
                              int border,
                              int HALO,
                              __local int* prev,
                              __local int* result,
                              __global int* outputBuffer, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[12];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 12; ++ocl_kernel_init_i) {
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

	int BLOCK_SIZE = get_local_size(0);
	int bx = get_group_id(0);
	int tx = get_local_id(0);

	// Each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size.
	int small_block_cols = BLOCK_SIZE - (iteration*HALO*2);

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkX = (small_block_cols*bx) - border;
	int blkXmax = blkX+BLOCK_SIZE-1;

	// calculate the global thread coordination
	int xidx = blkX+tx;

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
	
	int W = tx-1;
	int E = tx+1;

	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1))
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		prev[tx] = gpuSrc[xidx];
	}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

	
	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

	bool computed;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int i = 0; i < iteration || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++)
	{
private_ocl_kernel_loop_iter_counter[0]++;

		computed = false;
		
		if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid )
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

			computed = true;
			int left = prev[W];
			int up = prev[tx];
			int right = prev[E];
			int shortest = MIN(left, up);
			shortest = MIN(shortest, right);
			
			int index = cols*(startStep+i)+xidx;
			result[tx] = shortest + gpuWall[index];
			
			// ===================================================================
			// add debugging info to the debug output buffer...
			if (tx==11 && i==0)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

				// set bufIndex to what value/range of values you want to know.
				int bufIndex = gpuSrc[xidx];
				// dont touch the line below.
				outputBuffer[bufIndex] = 1;
			}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

			// ===================================================================
		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}


		OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);

		if(i==iteration-1)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

			// we are on the last iteration, and thus don't need to 
			// compute for the next step.
			break;
		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}


		if(computed)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

			//Assign the computation range
			prev[tx] = result[tx];
		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}

		OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);
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

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on "computed"
	if (computed)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

		gpuResults[xidx] = result[tx];
	}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 12; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}




