#include "ScaledGaussianPdf.hh"
//#include <limits>

EXEC_TARGET fptype device_ScaledGaussian (fptype* evt, fptype* p, unsigned int* indices) {
  int idx[4];
  idx[0] = indices[1];
  idx[1] = indices[2];
  idx[2] = indices[3];
  idx[3] = indices[4];

  fptype pidx[4];
  pidx[0] = p[idx[0]];
  pidx[1] = p[idx[1]];
  pidx[2] = p[idx[2]];
  pidx[3] = p[idx[3]];
  
  fptype x = evt[0]; 
  fptype mean = pidx[0] + pidx[2];
  fptype sigma = pidx[1] * (1 + pidx[3]);
  fptype ret = EXP(-0.5*(x-mean)*(x-mean)/(sigma*sigma));

#ifdef CUDAPRINT
  //if ((0 == THREADIDX) && (0 == BLOCKIDX) && (callnumber < 10)) 
    //cuPrintf("device_ScaledGaussian %f %i %i %f %f %i %p %f\n", x, indices[1], indices[2], mean, sigma, callnumber, indices, ret); 
#endif 
  //if ((gpuDebug & 1) && (0 == callnumber) && (THREADIDX == 6) && (0 == BLOCKIDX)) printf("[%i, %i] Scaled Gaussian: %f %f %f %f\n", BLOCKIDX, THREADIDX, x, mean, sigma, ret);

  return ret;
}

MEM_DEVICE device_function_ptr ptr_to_ScaledGaussian = device_ScaledGaussian; 

__host__ ScaledGaussianPdf::ScaledGaussianPdf (std::string n, Variable* _x, Variable* mean, Variable* sigma, Variable* delta, Variable* epsilon) 
: GooPdf(_x, n) 
{
  registerParameter(mean);
  registerParameter(sigma);
  registerParameter(delta);
  registerParameter(epsilon);

  std::vector<unsigned int> pindices;
  pindices.push_back(mean->getIndex());
  pindices.push_back(sigma->getIndex());
  pindices.push_back(delta->getIndex());
  pindices.push_back(epsilon->getIndex());
  GET_FUNCTION_ADDR(ptr_to_ScaledGaussian);
  initialise(pindices); 
}

