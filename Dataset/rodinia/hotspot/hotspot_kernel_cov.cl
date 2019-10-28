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

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define BLOCK_SIZE 16
__kernel void hotspot(  int iteration,  //number of iteration
                               global float *power,   //power input
                               global float *temp_src,    //temperature input/output
                               global float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset
							   int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[10];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 10; ++ocl_kernel_init_i) {
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


	local float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	local float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	local float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result

	float amb_temp = 80.0f;
	float step_div_Cap;
	float Rx_1,Ry_1,Rz_1;

	int bx = get_group_id(0);
	int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);

	step_div_Cap=step/Cap;

	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;

	// each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;
	int blkYmax = blkY+BLOCK_SIZE-1;
	int blkXmax = blkX+BLOCK_SIZE-1;

	// calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

	// load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
	int index = grid_cols*loadYidx+loadXidx;

	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

	int N = ty-1;
	int S = ty+1;
	int W = tx-1;
	int E = tx+1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int i=0; i<iteration || (private_ocl_kernel_loop_boundary_not_reached[0] = false) ; i++){
private_ocl_kernel_loop_iter_counter[0]++;

		computed = false;
		if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
		IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
		IN_RANGE(tx, validXmin, validXmax) && \
		IN_RANGE(ty, validYmin, validYmax) ) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

			computed = true;
			temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
			(temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
			(temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
			(amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
		OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);

		if(i==iteration-1)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
break;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

		if(computed)	 //Assign the computation range
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);
temp_on_cuda[ty][tx]= temp_t[ty][tx];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
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
	// small block perform the calculation and switch on ``computed''
	if (computed){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

	  temp_dst[index]= temp_t[ty][tx];
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 10; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
