#define DIVISIONS (1024)
float4 sortElem(float4 r, __local int* my_ocl_kernel_branch_triggered_recorder) {
	float4 nr;

	nr.x = (r.x > r.y) ? r.y : r.x;
	nr.y = (r.y > r.x) ? r.y : r.x;
	nr.z = (r.z > r.w) ? r.w : r.z;
	nr.w = (r.w > r.z) ? r.w : r.z;

	r.x = (nr.x > nr.z) ? nr.z : nr.x;
	r.y = (nr.y > nr.w) ? nr.w : nr.y;
	r.z = (nr.z > nr.x) ? nr.z : nr.x;
	r.w = (nr.w > nr.y) ? nr.w : nr.y;

	nr.x = r.x;
	nr.y = (r.y > r.z) ? r.z : r.y;
	nr.z = (r.z > r.y) ? r.z : r.y;
	nr.w = r.w;
	return nr;
}

float4 getLowest(float4 a, float4 b, __local int* my_ocl_kernel_branch_triggered_recorder)
{
	a.x = (a.x < b.w) ? a.x : b.w;
	a.y = (a.y < b.z) ? a.y : b.z;
	a.z = (a.z < b.y) ? a.z : b.y;
	a.w = (a.w < b.x) ? a.w : b.x;
	return a;
}

float4 getHighest(float4 a, float4 b, __local int* my_ocl_kernel_branch_triggered_recorder)
{
	b.x = (a.w >= b.x) ? a.w : b.x;
	b.y = (a.z >= b.y) ? a.z : b.y;
	b.z = (a.y >= b.z) ? a.y : b.z;
	b.w = (a.x >= b.w) ? a.x : b.w;
	return b;
}


__kernel void mergeSortFirst(__global float4 *input,__global float4 *result, const int listsize, __global int* ocl_kernel_branch_triggered_recorder){__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    int bx = get_group_id(0);

    if(bx*get_local_size(0) + get_local_id(0) < listsize/4){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        float4 r = input[bx*get_local_size(0)+ get_local_id(0)];
        result[bx * get_local_size(0) + get_local_id(0)] = sortElem(r, my_ocl_kernel_branch_triggered_recorder);
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
