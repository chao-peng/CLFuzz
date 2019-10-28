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





__kernel void trace_back2(__global char *str_npathflagp,
		__global char *str_nExtFlagp,
		__global int  *ndiffpos,
		__global char *instr1D,
		__global char *instr2D,
		__global char *outstr1,
		__global char *outstr2,
		__global MAX_INFO * strMaxInfop,
		int mfThreadNum, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[16];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 16; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[4];
bool private_ocl_kernel_loop_boundary_not_reached[4];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int i, j;
	int npos, maxPos, nlen;
	int npathflag;
	int nlaunchno;
	float maxScore;

	maxPos = 0;
	maxScore = strMaxInfop[0].fmaxscore;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (i = 1; i < mfThreadNum || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++)
	{
private_ocl_kernel_loop_iter_counter[0]++;

		if (maxScore < strMaxInfop[i].fmaxscore)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

			maxPos = i;
			maxScore = strMaxInfop[i].fmaxscore;
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
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

	npos = strMaxInfop[maxPos].nmaxpos;
	npathflag = str_npathflagp[npos] & 0x3;
	nlen = 0;

	i = strMaxInfop[maxPos].nposi;
	j = strMaxInfop[maxPos].nposj;
	nlaunchno = i + j;

	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while (1 || (private_ocl_kernel_loop_boundary_not_reached[1] = false))
	{
private_ocl_kernel_loop_iter_counter[1]++;

		if (npathflag == 3)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

			outstr1[nlen] = 23;
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			j--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno] - 1;
			nlaunchno--;
		}
		else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

if (npathflag == 1)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = 23;
			nlen++;
			i--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno];
			nlaunchno--;
		}
		else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

if (npathflag == 2)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			i--;
			j--;


			npos = npos - ndiffpos[nlaunchno] - ndiffpos[nlaunchno - 1] - 1;
			nlaunchno = nlaunchno - 2;
		}
		else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);


			return;
		}
}
}


		int nExtFlag = str_npathflagp[npos] / 4;
		if (npathflag == 3 && (nExtFlag == 2 || nExtFlag == 3))
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

			npathflag = 3;
		}

		else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);

if (npathflag == 1 && str_nExtFlagp[npos] == 1)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

			npathflag = 1;
		}
		else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);

			npathflag = str_npathflagp[npos] & 0x3;
		}
}

		if (i == 0 || j == 0)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

			break;
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}

		if (npathflag == PATH_END)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);

			break;
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
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

	i--;
	j--;

	private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
while(i >= 0 || (private_ocl_kernel_loop_boundary_not_reached[2] = false))
	{
private_ocl_kernel_loop_iter_counter[2]++;

		outstr1[nlen] = instr1D[i];
		outstr2[nlen] = 23;
		nlen++;
		i--;
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

	private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
while(j >= 0 || (private_ocl_kernel_loop_boundary_not_reached[3] = false))
	{
private_ocl_kernel_loop_iter_counter[3]++;

		outstr1[nlen] = 23;
		outstr2[nlen] = instr2D[j];
		nlen++;
		j--;
	}
if (private_ocl_kernel_loop_iter_counter[3] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 1);
}if (private_ocl_kernel_loop_iter_counter[3] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 2);
}if (private_ocl_kernel_loop_iter_counter[3] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[3]) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 8);
}

	strMaxInfop[0] = strMaxInfop[maxPos];
	strMaxInfop[0].noutputlen = nlen;

	return;
for (int update_recorder_i = 0; update_recorder_i < 16; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
