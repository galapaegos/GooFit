#ifndef RESONANCE_PDF_HH
#define RESONANCE_PDF_HH

#include "GooPdf.hh" 
#include "devcomplex.hh" 
typedef devcomplex<fptype> (*resonance_function_ptr) (const fptype &, const fptype &, const fptype &, const unsigned int*); 

EXEC_TARGET fptype twoBodyCMmom (const double &rMassSq, const fptype &d1m, const fptype &d2m);

EXEC_TARGET fptype dampingFactorSquare (const fptype &cmmom, const int& spin, const fptype &mRadius);

EXEC_TARGET fptype spinFactor (unsigned int spin, fptype motherMass, fptype daug1Mass, fptype daug2Mass, fptype daug3Mass, fptype m12, fptype m13, fptype m23, unsigned int cyclic_index);

class ResonancePdf : public GooPdf {
  // Service class intended to hold parametrisations of
  // resonances on Dalitz plots. Don't try to use this
  // as a standalone PDF! It should only be used as a
  // component in one of the friend classes. It extends
  // GooPdf so as to take advantage of the 
  // infrastructure, but will crash if used on its own. 

  friend class TddpPdf;
  friend class DalitzPlotPdf; 
  friend class IncoherentSumPdf; 
public:
  // Constructor for regular BW 
  ResonancePdf (string name, 
			  Variable* ar, 
			  Variable* ai, 
			  Variable* mass, 
			  Variable* width, 
			  unsigned int sp, 
			  unsigned int cyc); 

  // Gounaris-Sakurai
  ResonancePdf (string name, 
			  Variable* ar, 
			  Variable* ai, 
			  unsigned int sp, 
			  Variable* mass, 
			  Variable* width, 
			  unsigned int cyc); 
 
  // LASS constructor
  ResonancePdf (string name,
                          Variable* ar,
                          Variable* ai,
			  Variable* mass,
			  unsigned int sp,
                          Variable* width,
                          unsigned int cyc);
  

  // Nonresonant constructor
  ResonancePdf (string name, 
			  Variable* ar, 
			  Variable* ai);  

  // Gaussian constructor
  ResonancePdf (string name,
			  Variable* ar, 
			  Variable* ai,
			  Variable* mean, 
			  Variable* sigma,
			  unsigned int cyc);

private:
  void setConstantIndex (unsigned int idx) {host_indices[parameters + 1] = idx;}

  Variable* amp_real;
  Variable* amp_imag;
  /*
  Variable* mass;
  Variable* width;
  unsigned int spin;
  unsigned int cyclic_index;
  unsigned int eval_type;
  unsigned int resonance_type; 
  */ 
};

#endif
