#define DIVISIONS (1024)
float4 sortElem(float4 r, __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder) {
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

float4 getLowest(float4 a, float4 b, __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
	a.x = (a.x < b.w) ? a.x : b.w;
	a.y = (a.y < b.z) ? a.y : b.z;
	a.z = (a.z < b.y) ? a.z : b.y;
	a.w = (a.w < b.x) ? a.w : b.x;
	return a;
}

float4 getHighest(float4 a, float4 b, __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
	b.x = (a.w >= b.x) ? a.w : b.x;
	b.y = (a.z >= b.y) ? a.z : b.y;
	b.z = (a.y >= b.z) ? a.y : b.z;
	b.w = (a.x >= b.w) ? a.x : b.w;
	return b;
}



__kernel void
mergeSortPass(__global float4 *input, __global float4 *result,const int nrElems,int threadsPerDiv, __global int *constStartAddr, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[14];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 14; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	int gid = get_global_id(0);
	// The division to work on
	int division = gid / threadsPerDiv;
	if(division >= DIVISIONS) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
return;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

	// The block within the division
	int int_gid = gid - division * threadsPerDiv;
	int Astart = constStartAddr[division] + int_gid * nrElems;

	int Bstart = Astart + nrElems/2;
	global float4 *resStart;
    resStart= &(result[Astart]);

	if(Astart >= constStartAddr[division + 1])
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
return;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

	if(Bstart >= constStartAddr[division + 1]){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(int i=0; i<(constStartAddr[division + 1] - Astart) || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++)
		{
private_ocl_kernel_loop_iter_counter[0]++;

			resStart[i] = input[Astart + i];
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
		return;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

	int aidx = 0;
	int bidx = 0;
	int outidx = 0;
	float4 a, b;
	a = input[Astart + aidx];
	b = input[Bstart + bidx];

	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while(true || (private_ocl_kernel_loop_boundary_not_reached[1] = false))//aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
	{
private_ocl_kernel_loop_iter_counter[1]++;

		/**
		 * For some reason, it's faster to do the texture fetches here than
		 * after the merge
		 */
		float4 nextA = input[Astart + aidx + 1];
		float4 nextB = input[Bstart + bidx + 1];

		float4 na = getLowest(a,b, my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder);
		float4 nb = getHighest(a,b, my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder);
		a = sortElem(na, my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder);
		b = sortElem(nb, my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder);
		// Now, a contains the lowest four elements, sorted
		resStart[outidx++] = a;

		bool elemsLeftInA;
		bool elemsLeftInB;

		elemsLeftInA = (aidx + 1 < nrElems/2); // Astart + aidx + 1 is allways less than division border
		elemsLeftInB = (bidx + 1 < nrElems/2) && (Bstart + bidx + 1 < constStartAddr[division + 1]);

		if(elemsLeftInA){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

			if(elemsLeftInB){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

				if(nextA.x < nextB.x) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);
 aidx += 1; a = nextA; }
				else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
 bidx += 1;  a = nextB; }
			}
			else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);

				aidx += 1; a = nextA;
			}
		}
		else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);

			if(elemsLeftInB){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

				bidx += 1;  a = nextB;
			}
			else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);

				break;
			}
		}

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
	resStart[outidx++] = b;
for (int update_recorder_i = 0; update_recorder_i < 14; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
