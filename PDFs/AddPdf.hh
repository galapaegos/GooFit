#ifndef ADD_PDF_HH
#define ADD_PDF_HH

#include "GooPdf.hh" 

class AddPdf : public GooPdf {
public:

  AddPdf (std::string n, std::vector<Variable*> weights, std::vector<PdfBase*> comps); 
  AddPdf (std::string n, Variable* frac1, PdfBase* func1, PdfBase* func2); 
  virtual ~AddPdf ();
  __host__ virtual fptype normalise ();
  __host__ virtual bool hasAnalyticIntegral () const {return false;}
  __host__ virtual void recursiveSetIndices();

protected:
  __host__ virtual double sumOfNll (int numVars) const;

  std::vector<Variable*> m_weights;
  unsigned int weightIdx;

private:
  bool extended;
};

#endif
