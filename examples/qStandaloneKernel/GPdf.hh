#ifndef _G_PDF_FUNCTOR_HH
#define _G_PDF_FUNCTOR_HH

#ifdef TARGET_MPI
#include <mpi.h>
#endif
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include "thrust/iterator/constant_iterator.h" 
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <cassert> 
#include <set> 

#include "../../PdfBase.hh" 

EXEC_TARGET int dev_powi (int base, int exp); // Implemented in SmoothHistogramPdf.

#define CALLS_TO_PRINT 10 
typedef fptype (*device_function_ptr) (fptype*, fptype*, unsigned int*);            // Pass event, parameters, index into parameters. 
typedef fptype (*device_metric_ptr) (fptype, fptype*, unsigned int); 

extern void* host_fcn_ptr;

//class MetricTaker; 
class MetricTakerKnown;

class GPdf : public PdfBase { 
public:

  GPdf (Variable* x, std::string n, Variable *m, Variable *s);
  __host__ void setObservables(std::vector<Variable*> obs);
  __host__ virtual double calculateNLL () const;
  __host__ void evaluateAtPoints (std::vector<fptype>& points) const; 
  __host__ void evaluateAtPoints (Variable* var, std::vector<fptype>& res); 
  __host__ virtual fptype normalise () const;
  __host__ virtual fptype integrate (fptype lo, fptype hi) const {return 0;}
  __host__ virtual bool hasAnalyticIntegral () const {return false;} 
  __host__ fptype getValue (); 
  __host__ void getCompProbsAtDataPoints (std::vector<std::vector<fptype> >& values);
  __host__ void initialise (std::vector<unsigned int> pindices, void* dev_functionPtr = host_fcn_ptr); 
  __host__ void scan (Variable* var, std::vector<fptype>& values);
  __host__ virtual void setFitControl (FitControl* const fc, bool takeOwnerShip = true);
  __host__ virtual void setMetrics (); 
  __host__ void setParameterConstantness (bool constant = true);

  __host__ virtual void transformGrid (fptype* host_output); 
  static __host__ int findFunctionIdx (void* dev_functionPtr); 
  __host__ void setDebugMask (int mask, bool setSpecific = true) const; 

protected:
  __host__ virtual double sumOfNll (int numVars) const; 
  //MetricTaker* logger; 
  MetricTakerKnown *logger;
private:

};

class MetricTakerKnown : public thrust::unary_function<thrust::tuple<int, fptype*, int>, fptype>
{
public:
  MetricTakerKnown();
  
  EXEC_TARGET fptype operator() (thrust::tuple<int, fptype*, int> t) const;
  EXEC_TARGET fptype operator() (thrust::tuple<int, int, fptype*> t) const;
};

#endif