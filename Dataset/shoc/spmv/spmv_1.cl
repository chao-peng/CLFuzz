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
__kernel void
spmv_csr_scalar_kernel( __global const FPTYPE * restrict val,
#ifdef USE_TEXTURE
                        image2d_t vec,
#else
                        __global const FPTYPE * restrict vec,
#endif
                        __global const int * restrict cols,
                        __global const int * restrict rowDelimiters,
                       const int dim, __global FPTYPE * restrict out)
{
    int myRow = get_global_id(0);

    if (myRow < dim)
    {
        FPTYPE t=0;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++)
        {
            int col = cols[j];
#ifdef USE_TEXTURE
            t += val[j] * texFetch(vec,col);
#else
            t += val[j] * vec[col];
#endif
        }
        out[myRow] = t;
    }
}

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
