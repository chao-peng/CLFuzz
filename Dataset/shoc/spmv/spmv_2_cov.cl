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

//#define VECTOR_SIZE 32

#ifdef SINGLE_PRECISION
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif

#ifdef USE_TEXTURE
__constant sampler_t texFetchSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
FPTYPE texFetch(image2d_t image, const int idx) {
      int2 coord={idx%MAX_IMG_WIDTH,idx/MAX_IMG_WIDTH};
#ifdef SINGLE_PRECISION
        return read_imagef(image,texFetchSampler,coord).x;
#else
          return as_double2(read_imagei(image,texFetchSampler,coord)).x;
#endif
}
#endif

#define FPTYPE float


// ****************************************************************************
// Function: spmv_csr_scalar_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a thread per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   vec: dense vector for multiplication
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************

// ****************************************************************************
// Function: spmv_csr_vector_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a warp per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   vec: dense vector for multiplication
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   vecWidth: preferred simd width to use
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
__kernel void
spmv_csr_vector_kernel(__global const FPTYPE * restrict val,
#ifdef USE_TEXTURE
                       image2d_t vec,
#else
                       __global const FPTYPE * restrict vec,
#endif
                       __global const int * restrict cols,
                       __global const int * restrict rowDelimiters,
                       const int dim, const int vecWidth, __global FPTYPE * restrict out, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[6];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 6; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Thread ID in block
    int t = get_local_id(0);
    // Thread ID within warp
    int id = t & (vecWidth-1);
    // One row per warp
    int vecsPerBlock = get_local_size(0) / vecWidth;
    int myRow = (get_group_id(0) * vecsPerBlock) + (t / vecWidth);

    __local volatile FPTYPE partialSums[128];
    partialSums[t] = 0;

    if (myRow < dim)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        int vecStart = rowDelimiters[myRow];
        int vecEnd = rowDelimiters[myRow+1];
        FPTYPE mySum = 0;
        private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int j= vecStart + id; j < vecEnd || (private_ocl_kernel_loop_boundary_not_reached[0] = false);
             j+=vecWidth)
        {
private_ocl_kernel_loop_iter_counter[0]++;

            int col = cols[j];
#ifdef USE_TEXTURE
            mySum += val[j] * texFetch(vec,col);
#else
            mySum += val[j] * vec[col];
#endif
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

        partialSums[t] = mySum;
        OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

        // Reduce partial sums
	int bar = vecWidth / 2;
	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while(bar > 0 || (private_ocl_kernel_loop_boundary_not_reached[1] = false))
	{
private_ocl_kernel_loop_iter_counter[1]++;

	    if (id < bar) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
partialSums[t] += partialSums[t+ bar];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

	    OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
	    bar = bar / 2;
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

        // Write result
        if (id == 0)
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

            out[myRow] = partialSums[t];
        }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

// ****************************************************************************
// Function: spmv_ellpackr_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the ELLPACK-R data storage format; based on Vazquez et al (Univ. of
//   Almeria Tech Report 2009)
//
// Arguments:
//   val: array holding the non-zero values for the matrix in column
//   vec: dense vector for multiplication
//   major format and padded with zeros up to the length of longest row
//   cols: array of column indices for each element of the sparse matrix
//   rowLengths: array storing the length of each row of the sparse matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing directly
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
//
// Modifications:
//
// ****************************************************************************
