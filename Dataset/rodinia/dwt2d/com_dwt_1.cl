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
__kernel void c_CopySrcToComponents (__global int *d_r,
                                     __global int *d_g,
                                     __global int *d_b,
                                     __global unsigned char * cl_d_src,
                                     int pixels)
{
	int x = get_local_id(0);
	int gX= get_local_size(0) * get_group_id(0);

	__local unsigned char sData[THREADS*3];

    // Copy data to shared mem by 4bytes
    // other checks are not necessary, since
    // cl_d_src buffer is aligned to sharedDataSize
	sData[3 * x + 0] = cl_d_src [gX * 3 + 3 * x + 0];
	sData[3 * x + 1] = cl_d_src [gX * 3 + 3 * x + 1];
	sData[3 * x + 2] = cl_d_src [gX * 3 + 3 * x + 2];

	barrier(CLK_LOCAL_MEM_FENCE);

	int r, g, b;
	int offset = x*3;
	r = (int)(sData[offset]);
	g = (int)(sData[offset+1]);
	b = (int)(sData[offset+2]);

	int globalOutputPosition = gX + x;
	if (globalOutputPosition < pixels)
	{
		storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
	}

}


// Copy img src data into three separated component buffers
