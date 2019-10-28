#define THREADS 256
#define BOUNDARY_X 2


int divRndUp(int n,
             int d)
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
                     int pos)
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
                    int pos)
{
    d_c[pos] = c - 128;
}


// Copy img src data into three separated component buffers


// Copy img src data into three separated component buffers
__kernel void c_CopySrcToComponent (__global int *d_c,
									__global unsigned char * cl_d_src,
									int pixels)
{
	int x = get_local_id(0);
	int gX = get_local_size(0) * get_group_id(0);

	__local unsigned char sData[THREADS];

	sData[ x ] = cl_d_src [gX + x];

	barrier(CLK_LOCAL_MEM_FENCE);

	int c;

	c = (int) (sData[x]);

	int globalOutputPosition = gX + x;
	if (globalOutputPosition < pixels)
	{
		storeComponent(d_c, c, globalOutputPosition);
	}

}
