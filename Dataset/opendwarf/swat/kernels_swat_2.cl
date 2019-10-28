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
		int mfThreadNum)
{
	int i, j;
	int npos, maxPos, nlen;
	int npathflag;
	int nlaunchno;
	float maxScore;

	maxPos = 0;
	maxScore = strMaxInfop[0].fmaxscore;
	for (i = 1; i < mfThreadNum; i++)
	{
		if (maxScore < strMaxInfop[i].fmaxscore)
		{
			maxPos = i;
			maxScore = strMaxInfop[i].fmaxscore;
		}
	}

	npos = strMaxInfop[maxPos].nmaxpos;
	npathflag = str_npathflagp[npos] & 0x3;
	nlen = 0;

	i = strMaxInfop[maxPos].nposi;
	j = strMaxInfop[maxPos].nposj;
	nlaunchno = i + j;

	while (1)
	{
		if (npathflag == 3)
		{
			outstr1[nlen] = 23;
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			j--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno] - 1;
			nlaunchno--;
		}
		else if (npathflag == 1)
		{
			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = 23;
			nlen++;
			i--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno];
			nlaunchno--;
		}
		else if (npathflag == 2)
		{
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

			return;
		}


		int nExtFlag = str_npathflagp[npos] / 4;
		if (npathflag == 3 && (nExtFlag == 2 || nExtFlag == 3))
		{
			npathflag = 3;
		}

		else if (npathflag == 1 && str_nExtFlagp[npos] == 1)
		{
			npathflag = 1;
		}
		else
		{
			npathflag = str_npathflagp[npos] & 0x3;
		}

		if (i == 0 || j == 0)
		{
			break;
		}

		if (npathflag == PATH_END)
		{
			break;
		}
	}

	i--;
	j--;

	while(i >= 0)
	{
		outstr1[nlen] = instr1D[i];
		outstr2[nlen] = 23;
		nlen++;
		i--;
	}

	while(j >= 0)
	{
		outstr1[nlen] = 23;
		outstr2[nlen] = instr2D[j];
		nlen++;
		j--;
	}

	strMaxInfop[0] = strMaxInfop[maxPos];
	strMaxInfop[0].noutputlen = nlen;

	return;
}
