#include "macros.h"



__kernel void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     __global float* x, __global float* y, __global float* z,
	     __global float* Qr, __global float* Qi, __global struct kValues* ck)
{
  float sX;
  float sY;
  float sZ;
  float sQr;
  float sQi;

  // Determine the element of the X arrays computed by this thread
  int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + get_local_id(0);

  // Read block's X values from global mem to shared mem
  sX = x[xIndex];
  sY = y[xIndex];
  sZ = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  for (int kIndex = 0; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID); kIndex++) {
    if (kGlobalIndex + kIndex < numK) {
      float expArg;
      expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
      sQr = sQr + ck[kIndex].PhiMag * cos(expArg); // native_cos(expArg);
      sQi = sQi + ck[kIndex].PhiMag * sin(expArg); // native_sin(expArg);
    }
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
}
