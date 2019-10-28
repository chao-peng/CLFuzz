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

// Constants used in the MGVF computation
#define PI 3.14159f
#define ONE_OVER_PI (1.0f / PI)
#define MU 0.5f
#define LAMBDA (8.0f * MU + 1.0f)

// The number of threads per thread block
#define LOCAL_WORK_SIZE 256
#define NEXT_LOWEST_POWER_OF_TWO 256


// Regularized version of the Heaviside step function:
// He(x) = (atan(x) / pi) + 0.5
float heaviside(float x, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count, __local int* my_ocl_kernel_loop_recorder) {
	return (atan(x) * ONE_OVER_PI) + 0.5f;

	// A simpler, faster approximation of the Heaviside function
	/* float out = 0.0;
	if (x > -0.0001) out = 0.5;
	if (x >  0.0001) out = 1.0;
	return out; */
}


// Kernel to compute the Motion Gradient Vector Field (MGVF) matrix for multiple cells
__kernel void IMGVF_kernel(__global float *IMGVF_array, __global float *I_array, __constant int *I_offsets, int __constant *m_array,
						   __constant int *n_array, float vx, float vy, float e, int max_iterations, float cutoff, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[26];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 26; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[7];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 7; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[5];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 5; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[5];
bool private_ocl_kernel_loop_boundary_not_reached[5];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	
	// Shared copy of the matrix being computed
	__local float IMGVF[41 * 81];
	
	// Shared buffer used for two purposes:
	// 1) To temporarily store newly computed matrix values so that only
	//    values from the previous iteration are used in the computation.
	// 2) To store partial sums during the tree reduction which is performed
	//    at the end of each iteration to determine if the computation has converged.
	__local float buffer[LOCAL_WORK_SIZE];
	
	// Figure out which cell this thread block is working on
	int cell_num = get_group_id(0);
	
	// Get pointers to current cell's input image and inital matrix
	int I_offset = I_offsets[cell_num];
	__global float *IMGVF_global = &(IMGVF_array[I_offset]);
	__global float *I = &(I_array[I_offset]);
	
	// Get current cell's matrix dimensions
	int m = m_array[cell_num];
	int n = n_array[cell_num];
	
	// Compute the number of virtual thread blocks
	int max = (m * n + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;
	
	// Load the initial IMGVF matrix into shared memory
	int thread_id = get_local_id(0);
	int thread_block, i, j;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (thread_block = 0; thread_block < max || (private_ocl_kernel_loop_boundary_not_reached[0] = false); thread_block++) {
private_ocl_kernel_loop_iter_counter[0]++;

		int offset = thread_block * LOCAL_WORK_SIZE;
		i = (thread_id + offset) / n;
		j = (thread_id + offset) % n;
		if (i < m) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
IMGVF[(i * n) + j] = IMGVF_global[(i * n) + j];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

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
	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);
	
	// Set the converged flag to false
	__local int cell_converged;
	if (thread_id == 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
cell_converged = 0;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

	OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
	
	// Constants used to iterate through virtual thread blocks
	const float one_nth = 1.0f / (float) n;
	const int tid_mod = thread_id % n;
	const int tbsize_mod = LOCAL_WORK_SIZE % n;
	
	// Constant used in the computation of Heaviside values
	float one_over_e = 1.0f / e;
	
	// Iteratively compute the IMGVF matrix until the computation has
	//  converged or we have reached the maximum number of iterations
	int iterations = 0;
	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while ((! cell_converged) && (iterations < max_iterations) || (private_ocl_kernel_loop_boundary_not_reached[1] = false)) {
private_ocl_kernel_loop_iter_counter[1]++;

	
		// The total change to this thread's matrix elements in the current iteration
		float total_diff = 0.0f;
		
		int old_i = 0, old_j = 0;
		j = tid_mod - tbsize_mod;
		
		// Iterate over virtual thread blocks
		private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for (thread_block = 0; thread_block < max || (private_ocl_kernel_loop_boundary_not_reached[2] = false); thread_block++) {
private_ocl_kernel_loop_iter_counter[2]++;

			// Store the index of this thread's previous matrix element
			//  (used in the buffering scheme below)
			old_i = i;
			old_j = j;
			
			// Determine the index of this thread's current matrix element 
			int offset = thread_block * LOCAL_WORK_SIZE;
			i = (thread_id + offset) * one_nth;
			j += tbsize_mod;
			if (j >= n) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
j -= n;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

			
			float new_val = 0.0f, old_val = 0.0f;
			
			// Make sure the thread has not gone off the end of the matrix
			if (i < m) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

				// Compute neighboring matrix element indices
				int rowU = (i == 0) ? 0 : i - 1;
				int rowD = (i == m - 1) ? m - 1 : i + 1;
				int colL = (j == 0) ? 0 : j - 1;
				int colR = (j == n - 1) ? n - 1 : j + 1;
				
				// Compute the difference between the matrix element and its eight neighbors
				old_val = IMGVF[(i * n) + j];
				float U  = IMGVF[(rowU * n) + j   ] - old_val;
				float D  = IMGVF[(rowD * n) + j   ] - old_val;
				float L  = IMGVF[(i    * n) + colL] - old_val;
				float R  = IMGVF[(i    * n) + colR] - old_val;
				float UR = IMGVF[(rowU * n) + colR] - old_val;
				float DR = IMGVF[(rowD * n) + colR] - old_val;
				float UL = IMGVF[(rowU * n) + colL] - old_val;
				float DL = IMGVF[(rowD * n) + colL] - old_val;
				
				// Compute the regularized heaviside value for these differences
				float UHe  = heaviside((U  *       -vy)  * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float DHe  = heaviside((D  *        vy)  * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float LHe  = heaviside((L  *  -vx     )  * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float RHe  = heaviside((R  *   vx     )  * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float URHe = heaviside((UR * ( vx - vy)) * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float DRHe = heaviside((DR * ( vx + vy)) * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float ULHe = heaviside((UL * (-vx - vy)) * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				float DLHe = heaviside((DL * (-vx + vy)) * one_over_e, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
				
				// Update the IMGVF value in two steps:
				// 1) Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
				//                                   URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
				new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
													 URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
				// 2) Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
				float vI = I[(i * n) + j];
				new_val -= ((1.0f / LAMBDA) * vI * (new_val - vI));
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
			
			// Save the previous virtual thread block's value (if it exists)
			if (thread_block > 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

				offset = (thread_block - 1) * LOCAL_WORK_SIZE;
				if (old_i < m) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);
IMGVF[(old_i * n) + old_j] = buffer[thread_id];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}

			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}
			if (thread_block < max - 1) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

				// Write the new value to the buffer
				buffer[thread_id] = new_val;
			} else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);

				// We've reached the final virtual thread block,
				//  so write directly to the matrix
				if (i < m) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);
IMGVF[(i * n) + j] = new_val;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}

			}
			
			// Keep track of the total change of this thread's matrix elements
			total_diff += fabs(new_val - old_val);
			
			// We need to synchronize between virtual thread blocks to prevent
			//  threads from writing the values from the buffer to the actual
			//  IMGVF matrix too early
			OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);
		}
if (private_ocl_kernel_loop_iter_counter[2] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 1);
}if (private_ocl_kernel_loop_iter_counter[2] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 2);
}if (private_ocl_kernel_loop_iter_counter[2] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[2]) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 8);
}
		
		// We need to compute the overall sum of the change at each matrix element
		//  by performing a tree reduction across the whole threadblock
		buffer[thread_id] = total_diff;
		OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE);
		
		// Account for thread block sizes that are not a power of 2
		if (thread_id >= NEXT_LOWEST_POWER_OF_TWO) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);

			buffer[thread_id - NEXT_LOWEST_POWER_OF_TWO] += buffer[thread_id];
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);
}
		OCL_NEW_BARRIER(4,CLK_LOCAL_MEM_FENCE);
		
		// Perform the tree reduction
		int th;
		private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
for (th = NEXT_LOWEST_POWER_OF_TWO / 2; th > 0 || (private_ocl_kernel_loop_boundary_not_reached[3] = false); th /= 2) {
private_ocl_kernel_loop_iter_counter[3]++;

			if (thread_id < th) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[18], 1);

				buffer[thread_id] += buffer[thread_id + th];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[19], 1);
}
			OCL_NEW_BARRIER(5,CLK_LOCAL_MEM_FENCE);
		}
if (private_ocl_kernel_loop_iter_counter[3] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 1);
}if (private_ocl_kernel_loop_iter_counter[3] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 2);
}if (private_ocl_kernel_loop_iter_counter[3] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[3]) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 8);
}
		
		// Figure out if we have converged
		if(thread_id == 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[20], 1);

			float mean = buffer[thread_id] / (float) (m * n);
			if (mean < cutoff) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[22], 1);

				// We have converged, so set the appropriate flag
				cell_converged = 1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[23], 1);
}
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[21], 1);
}
		
		// We need to synchronize to ensure that all threads
		//  read the correct value of the convergence flag
		OCL_NEW_BARRIER(6,CLK_LOCAL_MEM_FENCE);
		
		// Keep track of the number of iterations we have performed
		iterations++;
	}
if (private_ocl_kernel_loop_iter_counter[1] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 1);
}if (private_ocl_kernel_loop_iter_counter[1] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 2);
}if (private_ocl_kernel_loop_iter_counter[1] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[1]) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 8);
}
	
	// Save the final IMGVF matrix to global memory
	private_ocl_kernel_loop_iter_counter[4] = 0;
private_ocl_kernel_loop_boundary_not_reached[4] = true;
for (thread_block = 0; thread_block < max || (private_ocl_kernel_loop_boundary_not_reached[4] = false); thread_block++) {
private_ocl_kernel_loop_iter_counter[4]++;

		int offset = thread_block * LOCAL_WORK_SIZE;
		i = (thread_id + offset) / n;
		j = (thread_id + offset) % n;
		if (i < m) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[24], 1);
IMGVF_global[(i * n) + j] = IMGVF[(i * n) + j];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[25], 1);
}

	}
if (private_ocl_kernel_loop_iter_counter[4] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 1);
}if (private_ocl_kernel_loop_iter_counter[4] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 2);
}if (private_ocl_kernel_loop_iter_counter[4] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[4]) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 8);
}
	
	// if (thread_id == 0) IMGVF_global[0] = (float) iterations;
for (int update_recorder_i = 0; update_recorder_i < 26; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 5; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
