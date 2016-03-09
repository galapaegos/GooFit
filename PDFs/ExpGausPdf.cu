#include "ExpGausPdf.hh"

EXEC_TARGET fptype device_ExpGaus (fptype* evt, fptype* p, unsigned int* indices) {
  int idx[5];
  idx[0] = indices[0];
  idx[1] = indices[1];
  idx[2] = indices[2];
  idx[3] = indices[3];
  idx[4] = indices[2 + idx[0]];

  fptype x     = evt[idx[4]]; 
  fptype mean  = p[idx[1]];
  fptype sigma = p[idx[2]];
  fptype alpha = p[idx[3]];

  fptype ret = 0.5*alpha; 
  fptype exparg = ret * (2*mean + alpha*sigma*sigma - 2*x);
  fptype erfarg = (mean + alpha*sigma*sigma - x) / (sigma * 1.4142135623);

  ret *= EXP(exparg); 
  ret *= ERFC(erfarg); 

  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_ExpGaus = device_ExpGaus; 

ExpGausPdf::ExpGausPdf (std::string n, Variable* _x, Variable* mean, Variable* sigma, Variable* tau) 
  : GooPdf(_x, n)
{
  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(mean));
  pindices.push_back(registerParameter(sigma));
  pindices.push_back(registerParameter(tau));
  GET_FUNCTION_ADDR(ptr_to_ExpGaus);
  initialise(pindices); 
}


