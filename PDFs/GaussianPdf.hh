#ifndef GAUSSIAN_PDF_HH
#define GAUSSIAN_PDF_HH

#include "GooPdf.hh" 

class GaussianPdf : public GooPdf {
public:
  GaussianPdf (std::string n, Variable* _x, Variable* m, Variable* s); 
  virtual ~GaussianPdf ();

  __host__ virtual fptype integrate (fptype lo, fptype hi);
  __host__ virtual bool hasAnalyticIntegral () const {return true;} 

  __host__ virtual void recursiveSetIndices ();

private:
  Variable *m_pSigma;
};

#endif
