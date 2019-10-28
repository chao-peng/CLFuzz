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

#define THREADS 256
#define BOUNDARY_X 2


int divRndUp(int n,
             int d, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count)
{
    return (n / d) + ((n % d) ? 1 : 0);
}


/* Store 3 RGB float components */
/*void storeComponents(__global float *d_r, __global float *d_g, __global float *d_b, __global const float r, __global const float g, __global const float b, int pos)
{
    d_r[pos] = (r/255.0f) - 0.5f;
    d_g[pos] = (g/255.0f) - 0.5f;
    d_b[pos] = (b/255.0f) - 0.5f;
}
*/


// Store 3 RGB intege components
void storeComponents(__global int *d_r,
                     __global int *d_g,
                     __global int *d_b,
                     int r,
                     int g,
                     int b,
                     int pos, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
}


/* Store float component */
/*__kernel void storeComponent(__global float *d_c, __global const float c, int pos)
{
    d_c[pos] = (c/255.0f) - 0.5f;
}
*/


// Store integer component
void storeComponent(__global int *d_c,
                    const int c,
                    int pos, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count)
{
    d_c[pos] = c - 128;
}


// Copy img src data into three separated component buffers


// Copy img src data into three separated component buffers
__kernel void c_CopySrcToComponent (__global int *d_c,
									__global unsigned char * cl_d_src,
									int pixels, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int x = get_local_id(0);
	int gX = get_local_size(0) * get_group_id(0);

	__local unsigned char sData[THREADS];

	sData[ x ] = cl_d_src [gX + x];

	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

	int c;

	c = (int) (sData[x]);

	int globalOutputPosition = gX + x;
	if (globalOutputPosition < pixels)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		storeComponent(d_c, c, globalOutputPosition, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count);
	}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
