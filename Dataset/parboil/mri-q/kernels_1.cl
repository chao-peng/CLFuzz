#include "macros.h"

__kernel void
ComputePhiMag_GPU(__global float* phiR, __global float* phiI, __global float* phiMag, int numK) {
  int indexK = get_global_id(0);
  float real = indexK;
  float imag = indexK;
  if (indexK < numK) {
    /*float*/ real = phiR[indexK];
    /*float*/ imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}
