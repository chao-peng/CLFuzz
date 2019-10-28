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

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable

#define __THREAD_FENCE_USED__

typedef struct {
	int nposi, nposj;
	int nmaxpos;
	float fmaxscore;
	int noutputlen;
}   MAX_INFO;

#define PATH_END 0
#define COALESCED_OFFSET 32

//Simple GPU barrier, with atomicAdd used
//Input: Local thread idx in a block, goal value
void __barrier_opencl_lock_based(int localID, int goalValue, volatile __global int *g_mutexOpencl, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count, __local int* my_ocl_kernel_loop_recorder)
{
	int tid = localID;
	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#ifdef __THREAD_FENCE_USED__
	write_mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	//other options
	//read_mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	//mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif

	if (tid == 0)
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		atom_add(g_mutexOpencl, 1);
		mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    int private_ocl_kernel_loop_iter_counter[3];
    bool private_ocl_kernel_loop_boundary_not_reached[3];
		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while (atom_add(g_mutexOpencl,0) < goalValue || (private_ocl_kernel_loop_boundary_not_reached[0] = false))
		{
private_ocl_kernel_loop_iter_counter[0]++;
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
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

	OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

__kernel void  MatchStringGPUSync(__global char  *pathFlag,
		__global char  *extFlag,
		__global float *nGapDist,
		__global float *hGapDist,
		__global float *vGapDist,
		__global int   *diffPos,
		__global int   *threadNum,
		int            rowNum,
		int            columnNum,
		__global char  *seq1,
		__global char  *seq2,
		int            blosumWidth,
		float          openPenalty,
		float          extensionPenalty,
		__global MAX_INFO *maxInfo,
		__global float *blosum62D,
		volatile __global int *mutexMem, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[18];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 18; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[3];
bool private_ocl_kernel_loop_boundary_not_reached[3];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int npos, ntablepos, tid;
	int npreposngap, npreposhgap, npreposvgap;
	int nLocalID = get_local_id(0);
	int blockNum = get_num_groups(0);
	int blockSize = get_local_size(0);
	int totalThreadNum = blockSize * blockNum;
	int threadid = get_global_id(0);

	int launchNo;
	int launchNum = rowNum + columnNum - 1;
	int indexi1 = -1;
	int indexj1 = 0;
	int indexi, indexj;
	int startPos = 2 * COALESCED_OFFSET;
	int noffset = 0;

	float fdist;
	float fdistngap, fdisthgap, fdistvgap;
	float ext_dist;
	float fmaxdist;



	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (launchNo = 2; launchNo < launchNum || (private_ocl_kernel_loop_boundary_not_reached[1] = false); launchNo++)
	{
private_ocl_kernel_loop_iter_counter[1]++;

		if (launchNo < rowNum)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

			indexi1++;
		}
		else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

if (launchNo == rowNum)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

			indexi1++;
			noffset = 1;
		}
		else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

			indexj1++;
		}
}

		private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for (tid = threadid; tid < threadNum[launchNo] || (private_ocl_kernel_loop_boundary_not_reached[2] = false); tid += totalThreadNum)
		{
private_ocl_kernel_loop_iter_counter[2]++;

			indexi = indexi1 - tid;
			indexj = indexj1 + tid;

			npos = startPos + tid;

			npreposhgap = npos - diffPos[launchNo];
			npreposvgap = npreposhgap - 1;
			npreposngap = npreposvgap - diffPos[launchNo - 1];

			ntablepos = seq1[indexi] * blosumWidth + seq2[indexj];
			//ntablepos = seq1C[indexi] * blosumWidth + seq2C[indexj];
			fdist = blosum62D[ntablepos];

			fmaxdist = nGapDist[npreposngap];
			fdistngap = fmaxdist + fdist;

			ext_dist  = hGapDist[npreposhgap] - extensionPenalty;
			fdisthgap = nGapDist[npreposhgap] - openPenalty;

			if (fdisthgap <= ext_dist)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

				fdisthgap = ext_dist;
				extFlag[npreposhgap] = 1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}

			ext_dist  = vGapDist[npreposvgap] - extensionPenalty;
			fdistvgap = nGapDist[npreposvgap] - openPenalty;

			if (fdistvgap <= ext_dist)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

				fdistvgap = ext_dist;
				pathFlag[npreposvgap] += 8;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}

			fdistngap = (fdistngap < 0.0f) ? 0.0f : fdistngap;
			fdisthgap = (fdisthgap < 0.0f) ? 0.0f : fdisthgap;
			fdistvgap = (fdistvgap < 0.0f) ? 0.0f : fdistvgap;

			hGapDist[npos] = fdisthgap;
			vGapDist[npos] = fdistvgap;

			//priority 00, 01, 10
			if (fdistngap >= fdisthgap && fdistngap >= fdistvgap)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

				fmaxdist = fdistngap;
				pathFlag[npos] = 2;
			}
			else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);

if (fdisthgap >= fdistngap && fdisthgap >= fdistvgap)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

				fmaxdist = fdisthgap;
				pathFlag[npos] = 1;
			}
			else
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);

				fmaxdist = fdistvgap;
				pathFlag[npos] = 3;
			}
}

			nGapDist[npos] = fmaxdist;

			//Here, the maximum match distance is 0, which means
			//previous alignment is useless
			if (fmaxdist <= 0.00000001f)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);

				pathFlag[npos] = PATH_END;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}

			if (maxInfo[threadid].fmaxscore < fmaxdist)
			{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);

				maxInfo[threadid].nposi = indexi + 1;
				maxInfo[threadid].nposj = indexj + 1;
				maxInfo[threadid].nmaxpos = npos;
				maxInfo[threadid].fmaxscore = fmaxdist;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);
}
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

		//GPU synchronization
		__barrier_opencl_lock_based(nLocalID, (launchNo - 1) * blockNum, mutexMem, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);
		startPos += diffPos[launchNo + 1] + noffset;
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
for (int update_recorder_i = 0; update_recorder_i < 18; update_recorder_i++) {
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]);
}
for (int update_recorder_i = 0; update_recorder_i < 3; update_recorder_i++) {
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]);
}
}
