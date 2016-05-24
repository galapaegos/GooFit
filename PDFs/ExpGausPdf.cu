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

  fptype x2 = 2.0*x;
  fptype mean2 = 2.0*mean;
  fptype sigma14 = sigma*1.4142135623;
  fptype sigma2 = sigma*sigma;
  fptype halpha = 0.5*alpha;
  fptype sigma2alpha = alpha*sigma2;

  fptype ret = halpha;

  fptype meanx2 = mean2 - x2;
  fptype meanx  = mean - x;
  fptype i_sigma = 1.0/sigma14;
  fptype combo1 = sigma2alpha + meanx2;
  fptype combo2 = meanx + sigma2alpha;
  fptype expr = combo1*ret;
  fptype erfr = combo2*i_sigma;

  fptype exparg = EXP(expr);
  fptype erfarg = ERFC(erfr);

  ret *= exparg; 
  ret *= erfarg; 

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


