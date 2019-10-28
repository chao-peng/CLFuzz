
/**
 * This kernel performs bilateral filtering of the depth map and outputs the vertex map and normal map
*/



__kernel void measurement_normals(
	__global const int* width,
	__global const int* height,
	__global const float3* vmap,
	__global float3* nmap
)
{
	const int x = get_global_id(0);
    const int y = get_global_id(1);
	int w = *width;
	int h = *height;

	int idx = w * y + x;

   // // normal map compute
   float3 v1 = (x < (w-1))?vmap[w * y + x + 1]:vmap[w * y + x - 1];
   float3 v2 = (y < (h-1))?vmap[w * (y + 1) + x]:vmap[w * (y - 1) + x];

   //float3 normal_nan = {0,0,0};
   float3 v = vmap[idx];
   nmap[idx] = normalize( ( isnan(v1) || isnan(v2) || isnan(v) ) ? NAN : cross(v1 - v,  v2 - v));
}
