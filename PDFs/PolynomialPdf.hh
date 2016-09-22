#ifndef POLYNOMIAL_PDF_HH
#define POLYNOMIAL_PDF_HH

#include "GooPdf.hh" 

class PolynomialPdf : public GooPdf {
public:
  PolynomialPdf (std::string n, Variable* _x, std::vector<Variable*> weights, Variable* x0 = 0, unsigned int lowestDegree = 0); 
  PolynomialPdf (string n, vector<Variable*> obses, vector<Variable*> coeffs, vector<Variable*> offsets, unsigned int maxDegree); 
  virtual ~PolynomialPdf ();

  __host__ fptype integrate (fptype lo, fptype hi); 
  //__host__ virtual bool hasAnalyticIntegral () const {return (1 == observables.size());} 
  __host__ fptype getCoefficient (int coef);

  __host__ virtual void recursiveSetIndices ();

private:
  int polyType;
  Variable* center; 
};

#endif
