#include "../../GlobalCudaDefines.hh"
#include "GPdf.hh" 
#include "thrust/sequence.h" 
#include "thrust/iterator/constant_iterator.h" 
#include <fstream> 

// LANDAU pdf : algorithm from CERNLIB G110 denlan
// same algorithm is used in GSL

MEM_CONSTANT fptype p1[5] = {0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253};
MEM_CONSTANT fptype q1[5] = {1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063};

MEM_CONSTANT fptype p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211};
MEM_CONSTANT fptype q2[5] = {1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714};

MEM_CONSTANT fptype p3[5] = {0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101};
MEM_CONSTANT fptype q3[5] = {1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675};

MEM_CONSTANT fptype p4[5] = {0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186};
MEM_CONSTANT fptype q4[5] = {1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511};

MEM_CONSTANT fptype p5[5] = {1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910};
MEM_CONSTANT fptype q5[5] = {1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357};

MEM_CONSTANT fptype p6[5] = {1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109};
MEM_CONSTANT fptype q6[5] = {1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939};

MEM_CONSTANT fptype a1[3] = {0.04166666667,-0.01996527778, 0.02709538966};
MEM_CONSTANT fptype a2[2] = {-1.845568670,-4.284640743};

MEM_CONSTANT fptype mpv;
MEM_CONSTANT fptype sigma;

__device__ fptype device_Landau (fptype* evt, fptype* p, unsigned int* indices) {
  fptype x     = evt[0];

  if (sigma <= 0) return 0;

  fptype v = (x - mpv)/sigma;

  fptype u, ue, us, denlan;
  if (v < -5.5) {
    u   = EXP(v+1.0);
    if (u < 1e-10) return 0.0;
    ue  = EXP(-1/u);
    us  = SQRT(u);
    denlan = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
  }
  else if (v < -1) {
    u   = EXP(-v-1);
    denlan = EXP(-u)*SQRT(u)*
      (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
      (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
  }
  else if (v < 1) {
    denlan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
      (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
  }
  else if (v < 5) {
    denlan = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
      (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
  }
  else if (v < 12) {
    u   = 1/v;
    denlan = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
      (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
  }
  else if (v < 50) {
    u   = 1/v;
    denlan = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u)/
      (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
  }
  else if (v < 300) {
    u   = 1/v;
    denlan = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u)/
      (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
  }
  else {
    u   = 1/(v-v*std::log(v)/(v+1));
    denlan = u*u*(1+(a2[0]+a2[1]*u)*u);
  }
  return denlan/sigma;
}

MEM_DEVICE device_function_ptr ptr_to_Landau = device_Landau;

// These variables are either function-pointer related (thus specific to this implementation)
// or constrained to be in the CUDAglob translation unit by nvcc limitations; otherwise they 
// would be in PdfBase. 

// Device-side, translation-unit constrained. 
MEM_CONSTANT fptype cudaArray[maxParams];           // Holds device-side fit parameters. 
MEM_CONSTANT unsigned int paramIndices[maxParams];  // Holds functor-specific indices into cudaArray. Also overloaded to hold integer constants (ie parameters that cannot vary.) 
MEM_CONSTANT fptype functorConstants[maxParams];    // Holds non-integer constants. Notice that first entry is number of events. 
MEM_CONSTANT fptype normalisationFactors[maxParams]; 

// For debugging 
MEM_CONSTANT int callnumber; 
MEM_CONSTANT int gpuDebug; 
MEM_CONSTANT unsigned int debugParamIndex;
MEM_DEVICE int internalDebug1 = -1; 
MEM_DEVICE int internalDebug2 = -1; 
MEM_DEVICE int internalDebug3 = -1; 
int cpuDebug = 0; 
#ifdef PROFILING
MEM_DEVICE fptype timeHistogram[10000]; 
fptype host_timeHist[10000];
#endif 

// Function-pointer related. 
MEM_DEVICE void* device_function_table[200]; // Not clear why this cannot be MEM_CONSTANT, but it causes crashes to declare it so. 
void* host_function_table[200];
unsigned int num_device_functions = 0; 
map<void*, int> functionAddressToDeviceIndexMap; 

// For use in debugging memory issues
void printMemoryStatus (std::string file, int line) {
  size_t memfree = 0;
  size_t memtotal = 0; 
  SYNCH(); 
// Thrust 1.7 will make the use of THRUST_DEVICE_BACKEND an error
#if THRUST_DEVICE_BACKEND==THRUST_DEVICE_BACKEND_OMP || THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_OMP
#else
  cudaMemGetInfo(&memfree, &memtotal); 
#endif
  SYNCH(); 
  std::cout << "Memory status " << file << " " << line << " Free " << memfree << " Total " << memtotal << " Used " << (memtotal - memfree) << std::endl;
}


#include <execinfo.h>
void* stackarray[10];
void abortWithCudaPrintFlush (std::string file, int line, std::string reason, const PdfBase* pdf = 0) {
#ifdef CUDAPRINT
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif
  std::cout << "Abort called from " << file << " line " << line << " due to " << reason << std::endl; 
  if (pdf) {
    PdfBase::parCont pars;
    pdf->getParameters(pars);
    std::cout << "Parameters of " << pdf->getName() << " : \n";
    for (PdfBase::parIter v = pars.begin(); v != pars.end(); ++v) {
      if (0 > (*v)->index) continue; 
      std::cout << "  " << (*v)->name << " (" << (*v)->index << ") :\t" << host_params[(*v)->index] << std::endl;
    }
  }

  std::cout << "Parameters (" << totalParams << ") :\n"; 
  for (int i = 0; i < totalParams; ++i) {
    std::cout << host_params[i] << " ";
  }
  std::cout << std::endl; 


  // get void* pointers for all entries on the stack
  size_t size = backtrace(stackarray, 10);
  // print out all the frames to stderr
  backtrace_symbols_fd(stackarray, size, 2);

  exit(1); 
}

/*
EXEC_TARGET fptype calculateEval (fptype rawPdf, fptype* evtVal, unsigned int par) {
  // Just return the raw PDF value, for use in (eg) normalisation. 
  return rawPdf; 
}

EXEC_TARGET fptype calculateNLL (fptype rawPdf, fptype* evtVal, unsigned int par) {
  //if ((10 > callnumber) && (THREADIDX < 10) && (BLOCKIDX == 0)) cuPrintf("calculateNll %i %f %f %f\n", callnumber, rawPdf, normalisationFactors[par], rawPdf*normalisationFactors[par]);
  //if (THREADIDX < 50) printf("Thread %i %f %f\n", THREADIDX, rawPdf, normalisationFactors[par]); 
  rawPdf *= normalisationFactors[par];
  return rawPdf > 0 ? -LOG(rawPdf) : 0; 
}

EXEC_TARGET fptype calculateProb (fptype rawPdf, fptype* evtVal, unsigned int par) {
  // Return probability, ie normalised PDF value.
  return rawPdf * normalisationFactors[par];
}

EXEC_TARGET fptype calculateBinAvg (fptype rawPdf, fptype* evtVal, unsigned int par) {
  rawPdf *= normalisationFactors[par];
  rawPdf *= evtVal[1]; // Bin volume 
  // Log-likelihood of numEvents with expectation of exp is (-exp + numEvents*ln(exp) - ln(numEvents!)). 
  // The last is constant, so we drop it; and then multiply by minus one to get the negative log-likelihood. 
  if (rawPdf > 0) {
    fptype expEvents = functorConstants[0]*rawPdf;
    return (expEvents - evtVal[0]*log(expEvents)); 
  }
  return 0; 
}

EXEC_TARGET fptype calculateBinWithError (fptype rawPdf, fptype* evtVal, unsigned int par) {
  // In this case interpret the rawPdf as just a number, not a number of events. 
  // Do not divide by integral over phase space, do not multiply by bin volume, 
  // and do not collect 200 dollars. evtVal should have the structure (bin entry, bin error). 
  //printf("[%i, %i] ((%f - %f) / %f)^2 = %f\n", BLOCKIDX, THREADIDX, rawPdf, evtVal[0], evtVal[1], POW((rawPdf - evtVal[0]) / evtVal[1], 2)); 
  rawPdf -= evtVal[0]; // Subtract observed value.
  rawPdf /= evtVal[1]; // Divide by error.
  rawPdf *= rawPdf; 
  return rawPdf; 
}

EXEC_TARGET fptype calculateChisq (fptype rawPdf, fptype* evtVal, unsigned int par) {
  rawPdf *= normalisationFactors[par];
  rawPdf *= evtVal[1]; // Bin volume 

  return pow(rawPdf * functorConstants[0] - evtVal[0], 2) / (evtVal[0] > 1 ? evtVal[0] : 1); 
}

MEM_DEVICE device_metric_ptr ptr_to_Eval         = calculateEval; 
MEM_DEVICE device_metric_ptr ptr_to_NLL          = calculateNLL;  
MEM_DEVICE device_metric_ptr ptr_to_Prob         = calculateProb; 
MEM_DEVICE device_metric_ptr ptr_to_BinAvg       = calculateBinAvg;  
MEM_DEVICE device_metric_ptr ptr_to_BinWithError = calculateBinWithError;
MEM_DEVICE device_metric_ptr ptr_to_Chisq        = calculateChisq; 
*/

void* host_fcn_ptr = 0;

/*
void* getMetricPointer (std::string name) {
  #define CHOOSE_PTR(ptrname) if (name == #ptrname) GET_FUNCTION_ADDR(ptrname);
  host_fcn_ptr = 0; 
  CHOOSE_PTR(ptr_to_Eval); 
  CHOOSE_PTR(ptr_to_NLL); 
  CHOOSE_PTR(ptr_to_Prob); 
  CHOOSE_PTR(ptr_to_BinAvg); 
  CHOOSE_PTR(ptr_to_BinWithError); 
  CHOOSE_PTR(ptr_to_Chisq); 

  assert(host_fcn_ptr); 

  return host_fcn_ptr;
#undef CHOOSE_PTR
}
*/

GPdf::GPdf (Variable* x, std::string n, Variable *m, Variable *s) 
  : PdfBase(x, n)
  , logger(0)
{
  //std::cout << "Created " << n << std::endl; 

  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(m));
  pindices.push_back(registerParameter(s));
  GET_FUNCTION_ADDR(ptr_to_Landau);
  initialise(pindices);

  setMetrics ();
}

__host__ int GPdf::findFunctionIdx (void* dev_functionPtr) {
  // Code specific to function-pointer implementation 
  map<void*, int>::iterator localPos = functionAddressToDeviceIndexMap.find(dev_functionPtr); 
  if (localPos != functionAddressToDeviceIndexMap.end()) {
    return (*localPos).second; 
  }

  int fIdx = num_device_functions;   
  host_function_table[num_device_functions] = dev_functionPtr;
  functionAddressToDeviceIndexMap[dev_functionPtr] = num_device_functions; 
  num_device_functions++; 
  MEMCPY_TO_SYMBOL(device_function_table, host_function_table, num_device_functions*sizeof(void*), 0, cudaMemcpyHostToDevice); 

#ifdef PROFILING
  host_timeHist[fIdx] = 0; 
  MEMCPY_TO_SYMBOL(timeHistogram, host_timeHist, 10000*sizeof(fptype), 0);
#endif 

  return fIdx; 
}

__host__ void GPdf::initialise (std::vector<unsigned int> pindices, void* dev_functionPtr) {
  if (!fitControl) setFitControl(new UnbinnedNllFit()); 

  // MetricTaker must be created after PdfBase initialisation is done.
  PdfBase::initialiseIndices(pindices); 

  functionIdx = findFunctionIdx(dev_functionPtr); 
  setMetrics(); 
}

__host__ void GPdf::setDebugMask (int mask, bool setSpecific) const {
  cpuDebug = mask; 
#if THRUST_DEVICE_BACKEND==THRUST_DEVICE_BACKEND_OMP
  gpuDebug = cpuDebug;
  if (setSpecific) debugParamIndex = parameters; 
#else
  MEMCPY_TO_SYMBOL(gpuDebug, &cpuDebug, sizeof(int), 0, cudaMemcpyHostToDevice);
  if (setSpecific) MEMCPY_TO_SYMBOL(debugParamIndex, &parameters, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
#endif
} 

__host__ void GPdf::setMetrics () {
  //if (logger) delete logger;
  //logger = new MetricTaker(this, getMetricPointer(fitControl->getMetric()));  

  if (logger) delete logger;
  logger = new MetricTakerKnown();
}

__host__ double GPdf::sumOfNll (int numVars) const {
  //printf ("sumOfNll\n");
  static thrust::plus<double> cudaPlus;
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array); 
  double dummy = 0;

  //if (host_callnumber >= 2) abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " debug abort", this); 
  thrust::counting_iterator<int> eventIndex(0); 
  return thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
				  thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
				  *logger, dummy, cudaPlus);   
}

__host__ double GPdf::calculateNLL () const {
  if (cpuDebug & 1) std::cout << getName() << " entering calculateNLL (" << host_callnumber << ")" << std::endl; 

  //manually copy params into values?
  Variable *m = getParameterByName ("mpv");
  Variable *s = getParameterByName ("sigma");

  //printf ("m->mixValue:%f\n", m->mixValue);
  //printf ("s->mixValue:%f\n", s->mixValue);

  cudaMemcpyToSymbol (mpv, &m->mixValue, sizeof (fptype));
  cudaMemcpyToSymbol (sigma, &s->mixValue, sizeof(fptype));

  //MEMCPY_TO_SYMBOL(callnumber, &host_callnumber, sizeof(int)); 
  //int oldMask = cpuDebug; 
  //if (0 == host_callnumber) setDebugMask(0, false); 
  //std::cout << "Start norm " << getName() << std::endl;
  normalise();
  //std::cout << "Norm done\n"; 
  //if ((0 == host_callnumber) && (1 == oldMask)) setDebugMask(1, false); 

  
  //if (cpuDebug & 1) {
  //std::cout << "Norm factors: ";
  //for (int i = 0; i < totalParams; ++i) std::cout << host_normalisation[i] << " ";
  //std::cout << std::endl;
  //} 
   
  if (host_normalisation[parameters] <= 0) 
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " non-positive normalisation", this);

  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  SYNCH(); // Ensure normalisation integrals are finished

  int numVars = observables.size(); 
  if (fitControl->binnedFit()) {
    numVars += 2;
    numVars *= -1; 
  }

  fptype ret = sumOfNll(numVars); 
  if (0 == ret) abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " zero NLL", this); 
  //if (cpuDebug & 1) std::cout << "Full NLL " << host_callnumber << " : " << 2*ret << std::endl;

  //setDebugMask(0); 

  //if ((cpuDebug & 1) && (host_callnumber >= 1)) abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " debug abort", this); 
  return 2*ret; 
}

__host__ void GPdf::evaluateAtPoints (Variable* var, std::vector<fptype>& res) {
  // NB: This does not project correctly in multidimensional datasets, because all observables
  // other than 'var' will have, for every event, whatever value they happened to get set to last
  // time they were set. This is likely to be the value from the last event in whatever dataset  
  // you were fitting to, but at any rate you don't get the probability-weighted integral over
  // the other observables. 

  copyParams(); 
  normalise(); 

  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  UnbinnedDataSet tempdata(observables);

  double step = (var->upperlimit - var->lowerlimit) / var->numbins; 
  for (int i = 0; i < var->numbins; ++i) {
    var->value = var->lowerlimit + (i+0.5)*step;
    tempdata.addEvent(); 
  }
  setData(&tempdata);

  thrust::counting_iterator<int> eventIndex(0); 
  thrust::constant_iterator<int> eventSize(observables.size()); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array); 
  thrust::device_vector<fptype> results(var->numbins); 

/*
  //MetricTaker evalor(this, getMetricPointer("ptr_to_Eval")); 
#ifdef TARGET_MPI
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + m_iEventsPerTask, arrayAddress, eventSize)),
		    results.begin(),
		    evalor); 
#else
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
		    results.begin(),
		    evalor); 
#endif
  thrust::host_vector<fptype> h_results = results;
  res.clear();
  res.resize(var->numbins);
  for (int i = 0; i < var->numbins; ++i) {
    res[i] = h_results[i] * host_normalisation[parameters];
  }
*/
}

__host__ void GPdf::evaluateAtPoints (std::vector<fptype>& points) const {
  /*
  std::set<Variable*> vars;
  getParameters(vars);
  unsigned int maxIndex = 0;
  for (std::set<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
    if ((*i)->getIndex() < maxIndex) continue;
    maxIndex = (*i)->getIndex();
  }
  std::vector<double> params;
  params.resize(maxIndex+1);
  for (std::set<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
    if (0 > (*i)->getIndex()) continue;
    params[(*i)->getIndex()] = (*i)->value;
  } 
  copyParams(params); 

  thrust::device_vector<fptype> d_vec = points; 
  normalise(); 
  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), *evalor);
  thrust::host_vector<fptype> h_vec = d_vec;
  for (unsigned int i = 0; i < points.size(); ++i) points[i] = h_vec[i]; 
  */
}

__host__ void GPdf::scan (Variable* var, std::vector<fptype>& values) {
  fptype step = var->upperlimit;
  step -= var->lowerlimit;
  step /= var->numbins;
  values.clear(); 
  for (fptype v = var->lowerlimit + 0.5*step; v < var->upperlimit; v += step) {
    var->value = v;
    copyParams();
    fptype curr = calculateNLL(); 
    values.push_back(curr);
  }
}

__host__ void GPdf::setParameterConstantness (bool constant) {
  PdfBase::parCont pars; 
  getParameters(pars); 
  for (PdfBase::parIter p = pars.begin(); p != pars.end(); ++p) {
    (*p)->fixed = constant; 
  }
}

__host__ fptype GPdf::getValue () {
  // Returns the value of the PDF at a single point. 
  // Execute redundantly in all threads for OpenMP multiGPU case
  copyParams(); 
  normalise(); 
  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  UnbinnedDataSet point(observables); 
  point.addEvent(); 
  setData(&point); 

  thrust::counting_iterator<int> eventIndex(0); 
  thrust::constant_iterator<int> eventSize(observables.size()); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array); 
  thrust::device_vector<fptype> results(1); 

/*
  MetricTaker evalor(this, getMetricPointer("ptr_to_Eval"));
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + 1, arrayAddress, eventSize)),
		    results.begin(),
		    evalor); 
*/
  return results[0];
}

__host__ fptype GPdf::normalise () const {
  //if (cpuDebug & 1) std::cout << "Normalising " << getName() << " " << hasAnalyticIntegral() << " " << normRanges << std::endl;

  if (!fitControl->metricIsPdf()) {
    host_normalisation[parameters] = 1.0; 
    return 1.0;
  }

  fptype ret = 1;
  if (hasAnalyticIntegral()) {
    for (obsConstIter v = obsCBegin(); v != obsCEnd(); ++v) { // Loop goes only over observables of this PDF. 
      //if (cpuDebug & 1) std::cout << "Analytically integrating " << getName() << " over " << (*v)->name << std::endl; 
      ret *= integrate((*v)->lowerlimit, (*v)->upperlimit);
    }
    host_normalisation[parameters] = 1.0/ret;
    //if (cpuDebug & 1) std::cout << "Analytic integral of " << getName() << " is " << ret << std::endl; 
    return ret; 
  } 

  int totalBins = 1; 
  for (obsConstIter v = obsCBegin(); v != obsCEnd(); ++v) {
    ret *= ((*v)->upperlimit - (*v)->lowerlimit);
    totalBins *= (integrationBins > 0 ? integrationBins : (*v)->numbins); 
    //if (cpuDebug & 1) std::cout << "Total bins " << totalBins << " due to " << (*v)->name << " " << integrationBins << " " << (*v)->numbins << std::endl; 
  }
  ret /= totalBins; 

  fptype dummy = 0; 
  static thrust::plus<fptype> cudaPlus;
  thrust::constant_iterator<fptype*> arrayAddress(normRanges); 
  thrust::constant_iterator<int> eventSize(observables.size());
  thrust::counting_iterator<int> binIndex(0); 
  fptype sum = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, arrayAddress)),
					thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, eventSize, arrayAddress)),
					*logger, dummy, cudaPlus); 

  //std::cout << "sum: " << sum << " mpv: " << getParameterByName("mpv")->value << " sigma: " << getParameterByName ("sigma")->value << std::endl;

  if (std::isnan(sum)) {
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " NaN in normalisation", this); 
  }
  else if (0 >= sum) { 
    abortWithCudaPrintFlush(__FILE__, __LINE__, "Non-positive normalisation", this); 
  }
  ret *= sum;
  if (0 == ret) abortWithCudaPrintFlush(__FILE__, __LINE__, "Zero integral"); 
  host_normalisation[parameters] = 1.0/ret;

  return (fptype) ret; 
}

#ifdef PROFILING
MEM_CONSTANT fptype conversion = (1.0 / CLOCKS_PER_SEC); 
EXEC_TARGET fptype callFunction (fptype* eventAddress, unsigned int functionIdx, unsigned int paramIdx) {
  clock_t start = clock();
  fptype ret = (*(reinterpret_cast<device_function_ptr>(device_function_table[functionIdx])))(eventAddress, cudaArray, paramIndices + paramIdx);
  clock_t stop = clock(); 
  if ((0 == THREADIDX + BLOCKIDX) && (stop > start)) {
    // Avoid issue when stop overflows and start doesn't. 
    timeHistogram[functionIdx*100 + paramIdx] += ((stop - start) * conversion); 
    //printf("Clock: %li %li %li | %u %f\n", (long) start, (long) stop, (long) (stop - start), functionIdx, timeHistogram[functionIdx]); 
  }

  return ret; 
}
#else 
EXEC_TARGET fptype callFunction (fptype* eventAddress, unsigned int functionIdx, unsigned int paramIdx) {
  return (*(reinterpret_cast<device_function_ptr>(device_function_table[functionIdx])))(eventAddress, cudaArray, paramIndices + paramIdx);
}
#endif 

// Notice that operators are distinguished by the order of the operands,
// and not otherwise! It's up to the user to make his tuples correctly. 

// Main operator: Calls the PDF to get a predicted value, then the metric 
// to get the goodness-of-prediction number which is returned to MINUIT. 
/*
EXEC_TARGET fptype MetricTaker::operator () (thrust::tuple<int, fptype*, int> t) const
{
  // Calculate event offset for this thread. 
  int eventIndex = thrust::get<0>(t);
  int eventSize  = thrust::get<2>(t);
  fptype* eventAddress = thrust::get<1>(t) + (eventIndex * abs(eventSize)); 

  // Causes stack size to be statically undeterminable.
  fptype ret = callFunction(eventAddress, functionIdx, parameters);

  // Notice assumption here! For unbinned fits the 'eventAddress' pointer won't be used
  // in the metric, so it doesn't matter what it is. For binned fits it is assumed that
  // the structure of the event is (obs1 obs2... binentry binvolume), so that the array
  // passed to the metric consists of (binentry binvolume). 
  ret = (*(reinterpret_cast<device_metric_ptr>(device_function_table[metricIndex])))(ret, eventAddress + (abs(eventSize)-2), parameters);
  return ret; 
}
*/

EXEC_TARGET fptype MetricTakerKnown::operator () (thrust::tuple<int, fptype*, int> t) const
{
  int eventIndex = thrust::get<0>(t);
  int eventSize = thrust::get<2>(t);

  fptype *eventAddress = thrust::get<1>(t) + (eventIndex * abs(eventSize));

  fptype *functionIdx = NULL;
  unsigned int *pIdx = NULL;
  fptype ret = device_Landau(eventAddress, functionIdx, pIdx);

  fptype r2 = ret *= normalisationFactors[0];
  
  return r2 > 0 ? -LOG(r2) : 0.0;
}
 
// Operator for binned evaluation, no metric. 
// Used in normalisation. 
/*
#define MAX_NUM_OBSERVABLES 5
EXEC_TARGET fptype MetricTaker::operator () (thrust::tuple<int, int, fptype*> t) const {
  // Bin index, event size, base address [lower, upper, numbins] 
 
  int evtSize = thrust::get<1>(t);
  int binNumber = thrust::get<0>(t);
  
  // Do not understand why this cannot be declared __shared__. Dynamically allocating shared memory is apparently complicated. 
  //fptype* binCenters = (fptype*) malloc(evtSize * sizeof(fptype));
  MEM_SHARED fptype binCenters[1024*MAX_NUM_OBSERVABLES];

  // To convert global bin number to (x,y,z...) coordinates: For each dimension, take the mod 
  // with the number of bins in that dimension. Then divide by the number of bins, in effect
  // collapsing so the grid has one fewer dimension. Rinse and repeat. 
  unsigned int* indices = paramIndices + parameters;
  for (int i = 0; i < evtSize; ++i) {
    fptype lowerBound = thrust::get<2>(t)[3*i+0];
    fptype upperBound = thrust::get<2>(t)[3*i+1];
    int numBins    = (int) FLOOR(thrust::get<2>(t)[3*i+2] + 0.5); 
    int localBin = binNumber % numBins;

    fptype x = upperBound - lowerBound; 
    x /= numBins;
    x *= (localBin + 0.5); 
    x += lowerBound;
    binCenters[indices[indices[0] + 2 + i]+THREADIDX*MAX_NUM_OBSERVABLES] = x; 
    binNumber /= numBins;
  }

  // Causes stack size to be statically undeterminable.
  fptype ret = callFunction(binCenters+THREADIDX*MAX_NUM_OBSERVABLES, functionIdx, parameters); 
  return ret; 
}
*/

EXEC_TARGET fptype MetricTakerKnown::operator () (thrust::tuple<int, int, fptype*> t) const
{
  int evtSize = thrust::get<1>(t);
  int binNumber = thrust::get<0> (t);

  MEM_SHARED fptype binCenters[1024*5];

  for (int i = 0; i < evtSize; ++i)
  {
    fptype lowerBound = thrust::get<2>(t)[3*i+0];
    fptype upperBound = thrust::get<2>(t)[3*i+1];
    int numBins    = (int) FLOOR(thrust::get<2>(t)[3*i+2] + 0.5);
    int localBin = binNumber % numBins;

    fptype x = upperBound - lowerBound;
    x /= numBins;
    x *= (localBin + 0.5);
    x += lowerBound;
    binCenters[THREADIDX*5] = x;
    binNumber /= numBins;
  }

  return device_Landau(binCenters + THREADIDX*5, 0, 0);
}

  //Untested so far, no examples call this function!
__host__ void GPdf::getCompProbsAtDataPoints (std::vector<std::vector<fptype> >& values)
{
  copyParams(); 
  double overall = normalise();
  std::cout << "normalize () - " << overall << std::endl;
  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  int numVars = observables.size(); 
  if (fitControl->binnedFit()) {
    numVars += 2;
    numVars *= -1; 
  }

  thrust::device_vector<fptype> results(numEntries); 
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array); 
  thrust::counting_iterator<int> eventIndex(0); 

/*
  MetricTaker evalor(this, getMetricPointer("ptr_to_Prob")); 

#ifdef TARGET_MPI
  //write to results, send to all?
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + m_iEventsPerTask, arrayAddress, eventSize)),
		    results.begin(), 
		    evalor);

  values.clear(); 
  values.resize(components.size() + 1);

  thrust::host_vector<fptype> host_results = results;
  for (unsigned int i = 0; i < host_results.size(); ++i) {
    values[0].push_back(host_results[i]);
  }
#else
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
		    results.begin(), 
		    evalor); 
  values.clear(); 
  values.resize(components.size() + 1);

  thrust::host_vector<fptype> host_results = results;
  for (unsigned int i = 0; i < host_results.size(); ++i) {
    values[0].push_back(host_results[i]);
  }
#endif 
*/

/*
  for (unsigned int i = 0; i < components.size(); ++i) {
    MetricTaker compevalor(components[i], getMetricPointer("ptr_to_Prob")); 
    thrust::counting_iterator<int> ceventIndex(0);
#ifdef TARGET_MPI 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(ceventIndex, arrayAddress, eventSize)),
		      thrust::make_zip_iterator(thrust::make_tuple(ceventIndex + m_iEventsPerTask, arrayAddress, eventSize)),
		      results.begin(), 
		      compevalor); 

    host_results = results;
    for (unsigned int j = 0; j < host_results.size(); ++j) {
      values[1 + i].push_back(host_results[j]); 
    }
#else
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(ceventIndex, arrayAddress, eventSize)),
		      thrust::make_zip_iterator(thrust::make_tuple(ceventIndex + numEntries, arrayAddress, eventSize)),
		      results.begin(), 
		      compevalor); 

    host_results = results;
    for (unsigned int j = 0; j < host_results.size(); ++j) {
      values[1 + i].push_back(host_results[j]); 
    }
#endif
  }
*/
}

// still need to add OpenMP/multi-GPU code here
__host__ void GPdf::transformGrid (fptype* host_output) { 
  generateNormRange(); 
  //normalise(); 
  int totalBins = 1; 
  for (obsConstIter v = obsCBegin(); v != obsCEnd(); ++v) {
    totalBins *= (*v)->numbins; 
  }

  thrust::constant_iterator<fptype*> arrayAddress(normRanges); 
  thrust::constant_iterator<int> eventSize(observables.size());
  thrust::counting_iterator<int> binIndex(0); 
  thrust::device_vector<fptype> d_vec;
  d_vec.resize(totalBins); 

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, arrayAddress)),
		    thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, eventSize, arrayAddress)),
		    d_vec.begin(), 
		    *logger); 

  //Extra copy here, fixme later
  thrust::host_vector<fptype> h_vec = d_vec;
  for (unsigned int i = 0; i < totalBins; ++i) host_output[i] = h_vec[i]; 
}

MetricTakerKnown::MetricTakerKnown()
{
}

/*
MetricTaker::MetricTaker (PdfBase* dat, void* dev_functionPtr) 
  : metricIndex(0)
  , functionIdx(dat->getFunctionIndex())
  , parameters(dat->getParameterIndex())
{
  //std::cout << "MetricTaker constructor with " << functionIdx << std::endl; 

  map<void*, int>::iterator localPos = functionAddressToDeviceIndexMap.find(dev_functionPtr); 
  if (localPos != functionAddressToDeviceIndexMap.end()) {
    metricIndex = (*localPos).second; 
  }
  else {
    metricIndex = num_device_functions; 
    host_function_table[num_device_functions] = dev_functionPtr;
    functionAddressToDeviceIndexMap[dev_functionPtr] = num_device_functions; 
    num_device_functions++; 
    MEMCPY_TO_SYMBOL(device_function_table, host_function_table, num_device_functions*sizeof(void*), 0, cudaMemcpyHostToDevice); 
  }
}

MetricTaker::MetricTaker (int fIdx, int pIdx) 
  : metricIndex(0)
  , functionIdx(fIdx)
  , parameters(pIdx)
{
  // This constructor should only be used for binned evaluation, ie for integrals. 
}
*/

__host__ void GPdf::setFitControl (FitControl* const fc, bool takeOwnerShip) {
  for (unsigned int i = 0; i < components.size(); ++i) {
    components[i]->setFitControl(fc, false); 
  }

  if ((fitControl) && (fitControl->getOwner() == this)) {
    delete fitControl; 
  }
  fitControl = fc; 
  if (takeOwnerShip) {
    fitControl->setOwner(this); 
  }
  setMetrics();
}

#include "../../PdfBase.cu" 
