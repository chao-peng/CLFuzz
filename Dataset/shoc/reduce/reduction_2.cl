#ifdef SINGLE_PRECISION
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif
#define FPTYPE float


// Currently, CPUs on Snow Leopard only support a work group size of 1
// So, we have a separate version of the kernel which doesn't use
// local memory. This version is only used when the maximum
// supported local group size is 1.
__kernel void
reduceNoLocal(__global FPTYPE *g_idata, __global FPTYPE *g_odata,
       unsigned int n)
{
    FPTYPE sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += g_idata[i];
    }
    g_odata[0] = sum;
}
