#include "KinLimitBWPdf.hh"

EXEC_TARGET fptype getMomentum (fptype mass, fptype pimass, fptype d0mass) {
  //if (mass <= 0) return 0; 
  //double lambda = mass*mass - pimass*pimass - d0mass*d0mass;
  //lambda *= lambda;
  //lambda -= 4*pimass*pimass*d0mass*d0mass;
  //if (lambda <= 0) return 0; 
  //return SQRT(0.5*lambda/mass); 
  fptype mass2 = mass*mass;
  fptype pimass2 = pimass*pimass;
  fptype d0mass2 = d0mass*d0mass;
  fptype i_mass = 1.0/mass;
  bool massLT0 = (mass <= 0);
  fptype lambda = mass2 - pimass2 - d0mass2;
  lambda -= 4.0 * pimass2 * d0mass2;
  bool lambdaLT0 = (lambda <= 0);

  return (massLT0 && lambdaLT0) ? 0.0 : SQRT(0.5*lambda*i_mass);
}

EXEC_TARGET fptype bwFactor (fptype momentum) {
  // 2.56 = 1.6^2, comes from radius for spin-1 particle
  //return 1/SQRT(1.0 + 2.56 * momentum*momentum);
  return RSQRT(1.0 + 2.56*momentum*momentum);
}

EXEC_TARGET fptype device_KinLimitBW (fptype* evt, fptype* p, unsigned int* indices) {
  int idx[5];
  idx[0] = indices[0];
  idx[1] = indices[1];
  idx[2] = indices[2];
  idx[3] = indices[3];
  idx[4] = indices[2 + idx[0]];

  fptype mean  = p[idx[1]];
  fptype width = p[idx[2]];
  fptype d0mass = cudaArray[idx[3]+0]; 
  fptype pimass = cudaArray[idx[3]+1]; 
  fptype x = evt[idx[4]]; 

  mean += d0mass;
  fptype width2 = width*width;
  x += d0mass;
  
  fptype pUsingRealMass = getMomentum(mean, pimass, d0mass); 
  if (0 >= pUsingRealMass) return 0; 
  fptype pUsingX     = getMomentum(x, pimass, d0mass); 
  
  mean *= mean; 
  fptype x2 = x*x;
  fptype bwUsingRealMass = bwFactor(pUsingRealMass);
  fptype pXdivRealMass = pUsingX/pUsingRealMass;
  fptype bwUsingX = bwFactor(pUsingX);
  fptype mean_x2 = mean - x2;
  fptype pXdivRealMass3 = pXdivRealMass*pXdivRealMass*pXdivRealMass;
  fptype bwXdivRealMass = bwUsingX/bwUsingRealMass;
  fptype bwXdivRealMass2 = bwXdivRealMass*bwXdivRealMass;

  fptype phspfactor  = pXdivRealMass3 * bwXdivRealMass2; 
  fptype phspMassSq  = mean_x2*mean_x2;
  fptype width_factor = phspfactor*width;
  fptype factor_mean = phspfactor*mean*width2;
  fptype phspGammaSq = width_factor*width_factor; 

  fptype ret = factor_mean/(phspMassSq + mean*phspGammaSq); 

  //  if (gpuDebug & 1) printf("[%i, %i] KinLimitBW: %f %f %f %f %f\n", BLOCKIDX, THREADIDX, x, mean, width, d0mass, pimass, ret);
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_KinLimitBW = device_KinLimitBW; 

__host__ KinLimitBWPdf::KinLimitBWPdf (std::string n, Variable* _x, Variable* mean, Variable* width) 
: GooPdf(_x, n) 
{
  registerParameter(mean);
  registerParameter(width);

  std::vector<unsigned int> pindices;
  pindices.push_back(mean->getIndex());
  pindices.push_back(width->getIndex());
  pindices.push_back(registerConstants(2));
  setMasses(1.8645, 0.13957); 
  GET_FUNCTION_ADDR(ptr_to_KinLimitBW);
  initialise(pindices);
}

__host__ void KinLimitBWPdf::setMasses (fptype bigM, fptype smallM) {
  fptype constants[2];
  constants[0] = bigM;
  constants[1] = smallM;
  //MEMCPY_TO_SYMBOL(functorConstants, constants, 2*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
}
