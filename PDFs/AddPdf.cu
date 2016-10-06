#include "AddPdf.hh"

EXEC_TARGET fptype device_AddPdfs (const fptype* __restrict evt, const fptype* __restrict params, unsigned int *funcIdx, unsigned int* indices) {
  //addition example - numParameters is 5
  int numParameters = params[*indices]; 
  fptype ret = 0;
  fptype totalWeight = 0; 

  *funcIdx += 1;
  for (int i = 0; i < numParameters; i ++)
  {
    //totalWeight += p[indices[i + 2]]   (0.9)

    //
    fptype weight = params[*indices + i + 1];

    *indices += 7;

    int np = params[*indices];
    int obs = params[*indices + np + 1];
    int con = params[*indices + np + 1 + obs + 1];
    fptype norm = params[*indices + np + 1 + obs + 1 + con + 1 + 1];

    totalWeight += weight;
    //first call to callFunction is device_Gaussian
    fptype curr = callFunction(evt, params, funcIdx, indices); 

    ret += weight * curr * norm; 

    //if ((gpuDebug & 1) && (0 == THREADIDX) && (0 == BLOCKIDX)) 
    //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
    //printf("Add comp %i: %f * %f * %f = %f (%f)\n", i, weight, curr, normalisationFactors[indices[i+1]], weight*curr*normalisationFactors[indices[i+1]], ret); 

  }

  numParameters = params[*indices];
  int obs = params[*indices + numParameters + 1];
  int con = params[*indices + numParameters + 1 + obs + 1];
  fptype normFactors = params[*indices + numParameters + 1 + obs + 1 + con + 1 + 1];

  // numParameters does not count itself. So the array structure for two functions is
  // nP | F P w | F P
  // in which nP = 5. Therefore the parameter index for the last function pointer is nP, and the function index is nP-1. 
  //fptype last = (*(reinterpret_cast<device_function_ptr>(device_function_table[indices[numParameters-1]])))(evt, p, paramIndices + indices[numParameters]);

  //addition example calls device_Polynomial
  fptype last = callFunction(evt, params, funcIdx, indices);
  ret += (1 - totalWeight) * last * normFactors; 

  //if ((THREADIDX < 50) && (isnan(ret))) printf("NaN final component %f %f\n", last, totalWeight); 

  //if ((gpuDebug & 1) && (0 == THREADIDX) && (0 == BLOCKIDX)) 
  //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
  //printf("Add final: %f * %f * %f = %f (%f)\n", (1 - totalWeight), last, normalisationFactors[indices[numParameters]], (1 - totalWeight) *last* normalisationFactors[indices[numParameters]], ret); 
  
  return ret; 
}

EXEC_TARGET fptype device_AddPdfsExt (const fptype* __restrict evt, const fptype* __restrict params, unsigned int* funcIdx, unsigned int* indices) { 
  // numParameters does not count itself. So the array structure for two functions is
  // nP | F P w | F P w
  // in which nP = 6. 
  funcIdx += 1;

  int numParameters = indices[0]; 
  fptype ret = 0;
  fptype totalWeight = 0; 
  for (int i = 1; i < numParameters; i += 3) {    
    *indices += 3;
    //fptype curr = (*(reinterpret_cast<device_function_ptr>(device_function_table[indices[i]])))(evt, p, paramIndices + indices[i+1]);
    fptype curr = callFunction(evt, params, funcIdx, indices); 
    fptype weight = params[*indices + i+2];
    ret += weight * curr * params[*indices + i+1]; 

    totalWeight += weight; 
    //if ((gpuDebug & 1) && (THREADIDX == 0) && (0 == BLOCKIDX)) 
    //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
    //printf("AddExt: %i %E %f %f %f %f %f %f\n", i, curr, weight, ret, totalWeight, normalisationFactors[indices[i+1]], evt[0], evt[8]);
  }
  ret /= totalWeight; 
  //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
  //if ((gpuDebug & 1) && (THREADIDX == 0) && (0 == BLOCKIDX)) 
  //printf("AddExt result: %f\n", ret); 
  
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_AddPdfs = device_AddPdfs; 
MEM_DEVICE device_function_ptr ptr_to_AddPdfsExt = device_AddPdfsExt; 

AddPdf::AddPdf (std::string n, std::vector<Variable*> weights, std::vector<PdfBase*> comps) 
  : GooPdf(0, n) 
  , extended(true)
{

  assert((weights.size() == comps.size()) || (weights.size() + 1 == comps.size())); 

  // Indices stores (function index)(function parameter index)(weight index) triplet for each component. 
  // Last component has no weight index unless function is extended. 
  for (std::vector<PdfBase*>::iterator p = comps.begin(); p != comps.end(); ++p) {
    components.push_back(*p); 
    assert(components.back()); 
  }

  getObservables(observables); 

  weightIdx = 0;

  std::vector<unsigned int> pindices;
  for (unsigned int w = 0; w < weights.size(); ++w) {
    assert(components[w]);
    //pindices.push_back(components[w]->getFunctionIndex());
    //pindices.push_back(components[w]->getParameterIndex());
    pindices.push_back(registerParameter(weights[w])); 
    m_weights.push_back (weights[w]);
  }
  assert(components.back()); 
  if (weights.size() < components.size()) {
    //pindices.push_back(components.back()->getFunctionIndex());
    //pindices.push_back(components.back()->getParameterIndex());
    extended = false; 
  }


  if (extended) GET_FUNCTION_ADDR(ptr_to_AddPdfsExt);
  else GET_FUNCTION_ADDR(ptr_to_AddPdfs);

  initialise(pindices); 
} 


AddPdf::AddPdf (std::string n, Variable* frac1, PdfBase* func1, PdfBase* func2) 
  : GooPdf(0, n) 
  , extended(false)
{
  // Special-case constructor for common case of adding two functions.
  components.push_back(func1);
  components.push_back(func2);
  getObservables(observables); 

  weightIdx = 0;

  std::vector<unsigned int> pindices;
  pindices.push_back(func1->getFunctionIndex());
  pindices.push_back(func1->getParameterIndex());
  pindices.push_back(registerParameter(frac1)); 

  pindices.push_back(func2->getFunctionIndex());
  pindices.push_back(func2->getParameterIndex());
    
  GET_FUNCTION_ADDR(ptr_to_AddPdfs);

  initialise(pindices); 
} 

AddPdf::~AddPdf ()
{
}

__host__ void AddPdf::recursiveSetIndices ()
{
  //(brad): all pdfs need to add themselves to device list so we can increment
  if (extended)
    GET_FUNCTION_ADDR(ptr_to_AddPdfsExt);
  else
    GET_FUNCTION_ADDR(ptr_to_AddPdfs);
  host_function_table[num_device_functions] = host_fcn_ptr;
  functionIdx = num_device_functions;
  num_device_functions ++;

  //(brad): Each call needs to set the indices into the array.  In otherwords, each function needs to 'pre-load'
  //all parameters.  This means that each function will need to read all their variables/parameters they will use
  //before calling any other functions.
  host_params[totalParams++] = parameterList.size ();
  parametersIdx = totalParams;
  for (int i = 0; i < parameterList.size (); i++)
    host_params[totalParams++] = parameterList[i]->value;

  host_params[totalParams++] = observables.size ();
  observablesIdx = totalParams;
  for (int i = 0; i < observables.size (); i++)
    host_params[totalParams++] = observables[i]->value;

  host_params[totalParams++] = constants.size ();
  constantsIdx = totalParams;
  for (int i = 0; i < constants.size (); i++)
    host_params[totalParams++] = constants[i];

  //normalisation
  host_params[totalParams++] = 1;
  normalisationIdx = totalParams;
  host_params[totalParams++] = 0;

  for (unsigned int i = 0; i < components.size(); ++i)
    components[i]->recursiveSetIndices();

  generateNormRange ();
}

__host__ fptype AddPdf::normalise (int stream) {
  //if (cpuDebug & 1)
  //std::cout << "Normalising AddPdf " << getName() << std::endl;

  fptype ret = 0;
  fptype totalWeight = 0; 

  //Unsure if I need to know the functions indices?
  for (unsigned int i = 0; i < components.size()-1; ++i)
  {
    //addition example:  This parameter is 0.9
    //fptype weight = host_params[host_indices[parameters + 3*(i+1)]]; 
    fptype weight = host_params[parametersIdx];
    totalWeight += weight;
    //curr is 2.506628274631
    fptype curr = components[i]->normalise(stream); 
    ret += curr*weight;
  }

  //last is 10
  fptype last = components.back()->normalise(stream); 
  if (extended) {
    //Whoah, must test this out.  This is where it *should* be
    fptype lastWeight = host_params[parametersIdx + + 3*components.size()];
    totalWeight += lastWeight;
    ret += last * lastWeight; 
    ret /= totalWeight; 
  }
  else {
    ret += (1 - totalWeight) * last;
  }

  //host_normalisation[parameters] = 1.0; 
  host_params[normalisationIdx] = 1.0;

  if (getSpecialMask() & PdfBase::ForceCommonNorm) {
    // Want to normalise this as 
    // (f1 A + (1-f1) B) / int (f1 A + (1-f1) B) 
    // instead of default 
    // (f1 A / int A) + ((1-f1) B / int B).

    for (unsigned int i = 0; i < components.size(); ++i) {
      host_normalisation[components[i]->getParameterIndex()] = (1.0 / ret);
    }
  }

  //if (cpuDebug & 1) std::cout << getName() << " integral returning " << ret << std::endl; 
  return ret; 
}

__host__ double AddPdf::sumOfNll (int stream, int numVars) const {
  static thrust::plus<double> cudaPlus;
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array); 
  //thrust::constant_iterator<fptype*> paramAddress(dev_param_array); 
  double dummy = 0;
  thrust::counting_iterator<int> eventIndex(0); 

#ifdef TARGET_MPI
  double r = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, paramAddress, eventSize)), 
					thrust::make_zip_iterator(thrust::make_tuple(eventIndex + m_iEventsPerTask, arrayAddress, paramAddress, eventSize)),
					*logger, dummy, cudaPlus);

  double ret;
  MPI_Allreduce(&r, &ret, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  double ret = 0.0;
  if (stream == 1)
  {
    thrust::constant_iterator<fptype*> paramAddress(dev_param_array_s1); 
    thrust::transform_reduce(thrust::cuda::par.on(m_stream1), thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, paramAddress, eventSize)), 
					thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, paramAddress, eventSize)),
					*logger, dummy, cudaPlus); 
  }
  else if (stream == 2)
  {
    thrust::constant_iterator<fptype*> paramAddress(dev_param_array_s2);
    thrust::transform_reduce(thrust::cuda::par.on(m_stream2), thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, paramAddress, eventSize)), 
					thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, paramAddress, eventSize)),
					*logger, dummy, cudaPlus); 
  }
#endif

  if (extended) {
    fptype expEvents = 0; 
    //std::cout << "Weights:"; 
    for (unsigned int i = 0; i < components.size(); ++i) {
      expEvents += host_params[host_indices[parameters + 3*(i+1)]]; 
      //std::cout << " " << host_params[host_indices[parameters + 3*(i+1)]]; 
    }
    // Log-likelihood of numEvents with expectation of exp is (-exp + numEvents*ln(exp) - ln(numEvents!)). 
    // The last is constant, so we drop it; and then multiply by minus one to get the negative log-likelihood. 
    ret += (expEvents - numEvents*log(expEvents)); 
    //std::cout << " " << expEvents << " " << numEvents << " " << (expEvents - numEvents*log(expEvents)) << std::endl; 
  }

  return ret; 
}
