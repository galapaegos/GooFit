#include "DalitzPlotPdf.hh"
#include <complex>
using std::complex; 

const int resonanceOffset_DP = 4; // Offset of the first resonance into the parameter index array 
// Offset is number of parameters, constant index, number of resonances (not calculable 
// from nP because we don't know what the efficiency might need), and cache index. Efficiency 
// parameters are after the resonance information. 

// The function of this array is to hold all the cached waves; specific 
// waves are recalculated when the corresponding resonance mass or width 
// changes. Note that in a multithread environment each thread needs its
// own cache, hence the '10'. Ten threads should be enough for anyone! 
MEM_DEVICE devcomplex<fptype>* cResonances[10]; 

EXEC_TARGET inline int parIndexFromResIndex_DP (int resIndex) {
  return resonanceOffset_DP + resIndex*resonanceSize; 
}

EXEC_TARGET devcomplex<fptype> device_DalitzPlot_calcIntegrals (fptype m12, fptype m13, int res_i, int res_j, unsigned int *funcIdx, unsigned int* indices) {
  // Calculates BW_i(m12, m13) * BW_j^*(m12, m13). 
  // This calculation is in a separate function so
  // it can be cached. Note that this function expects
  // to be called on a normalisation grid, not on 
  // observed points, that's why it doesn't use 
  // cResonances. No need to cache the values at individual
  // grid points - we only care about totals. 
  int numParams = cudaArray[*indices];

  fptype motherMass = cudaArray[*indices + 1]; 
  fptype daug1Mass  = cudaArray[*indices + 2]; 
  fptype daug2Mass  = cudaArray[*indices + 3]; 
  fptype daug3Mass  = cudaArray[*indices + 4];  

  int numObs = int (cudaArray[*indices + numParams + 1 + numObs + 1]);
  int obsIdx = numParams + 2;

  int numCons = int (cudaArray[*indices + numParams + 1 + numObs + 1]);
  int consIdx = numParams + 1 + numObs + 2;

  devcomplex<fptype> ret; 
  if (!inDalitz(m12, m13, motherMass, daug1Mass, daug2Mass, daug3Mass)) return ret;
  fptype m23 = motherMass*motherMass + daug1Mass*daug1Mass + daug2Mass*daug2Mass + daug3Mass*daug3Mass - m12 - m13; 

  unsigned int numResonances = (unsigned int)cudaArray[*indices + consIdx + 4];

  *funcIdx += 1;
  *indices += numParams + 1 + numObs + 1 + numCons + 3;

  //todo: (brad) we need to pass in the total offset for these
  //int parameter_i = parIndexFromResIndex_DP(res_i);
  unsigned int tmp_f_i = *funcIdx + res_i;
  unsigned int tmp_p_i = *indices;
  ret = getResonanceAmplitude(m12, m13, m23, &tmp_f_i, &tmp_p_i);

  //int parameter_j = parIndexFromResIndex_DP(res_j);
  int parameter_j = resonanceOffset_DP + res_j*numResonances;
  //unsigned int functn_j = cudaArray[*indices + consIdx + parameter_j + 3];
  //unsigned int params_j = cudaArray[*indices + consIdx + parameter_j + 4];
  unsigned int tmp_f_j = *funcIdx + res_j;
  unsigned int tmp_p_j = *indices;
  ret *= conj(getResonanceAmplitude(m12, m13, m23, &tmp_f_j, &tmp_p_j));

  *indices += numParams + 1 + numObs + 1 + numCons + 1 + 2;

  return ret; 
}

EXEC_TARGET fptype device_DalitzPlot (fptype* evt, unsigned int *funcIdx, unsigned int* indices) {
  int numParams = cudaArray[*indices];
 
  int paramIdx = 1;
  fptype m12 = cudaArray[*indices + paramIdx + 0]; 
  fptype m13 = cudaArray[*indices + paramIdx + 1];
  int evtNum = int (FLOOR(0.5 + cudaArray[*indices + paramIdx + 2]));

  int numObs = cudaArray[*indices + numParams + 1];
  int obsIdx = numParams + 2; 
  int numCons = int (cudaArray[*indices + numParams + 1 + numObs + 1]);
  int consIdx = numParams + 1 + numObs + 2;
  fptype motherMass = cudaArray[*indices + consIdx + 0]; 
  fptype daug1Mass  = cudaArray[*indices + consIdx + 1]; 
  fptype daug2Mass  = cudaArray[*indices + consIdx + 2]; 
  fptype daug3Mass  = cudaArray[*indices + consIdx + 3]; 

  if (!inDalitz(m12, m13, motherMass, daug1Mass, daug2Mass, daug3Mass)) return 0; 

  devcomplex<fptype> totalAmp(0, 0);

  unsigned int numResonances = (unsigned int)cudaArray[*indices + consIdx + 4];
  unsigned int cacheToUse    = (unsigned int)cudaArray[*indices + consIdx + 5]; 

#pragma unroll
  for (int i = 0; i < numResonances; ++i) {
    //debug this:
    //int paramIndex  = parIndexFromResIndex_DP(i);
    fptype amp_real = cudaArray[*indices + paramIdx + 4 + i*numResonances + 3];
    fptype amp_imag = cudaArray[*indices + paramIdx + 4 + i*numResonances + 4];

    devcomplex<fptype> matrixelement((cResonances[cacheToUse][evtNum*numResonances + i]).real,
				     (cResonances[cacheToUse][evtNum*numResonances + i]).imag); 
    matrixelement.multiply(amp_real, amp_imag); 
    totalAmp += matrixelement; 
  } 

  fptype ret = norm2(totalAmp); 

  *funcIdx += 1;
  // + 3, +1 for num constants, + 1 for num of norms, + 1 over the norms;
  *indices += numParams + 1 + numObs + 1 + numCons + 3;
  //we get our function index by adding
  //int effFunctionIdx = parIndexFromResIndex_DP(numResonances); 
  fptype eff = callFunction(evt, funcIdx, indices); 
  ret *= eff;

  //printf("DalitzPlot evt %i zero: %i %i %f (%f, %f).\n", evtNum, numResonances, effFunctionIdx, eff, totalAmp.real, totalAmp.imag); 

  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_DalitzPlot = device_DalitzPlot; 

__host__ DalitzPlotPdf::DalitzPlotPdf (std::string n, 
							   Variable* m12, 
							   Variable* m13, 
							   Variable* eventNumber, 
							   DecayInfo* decay, 
							   GooPdf* efficiency)
  : GooPdf(0, n) 
  , decayInfo(decay)
  , _m12(m12)
  , _m13(m13)
  , dalitzNormRange(0)
  , cachedWaves(0) 
  , integrals(0)
  , forceRedoIntegrals(true)
  , totalEventSize(3) // Default 3 = m12, m13, evtNum 
  , cacheToUse(0) 
  , integrators(0)
  , calculators(0) 
{
  registerObservable(_m12);
  registerObservable(_m13);
  registerObservable(eventNumber); 

  parameterList.push_back (_m12);
  parameterList.push_back (_m13);
  parameterList.push_back (eventNumber);

  fptype decayConstants[5];
  
  std::vector<unsigned int> pindices;
  pindices.push_back(registerConstants(5)); 
  decayConstants[0] = decayInfo->motherMass;
  decayConstants[1] = decayInfo->daug1Mass;
  decayConstants[2] = decayInfo->daug2Mass;
  decayConstants[3] = decayInfo->daug3Mass;
  decayConstants[4] = decayInfo->meson_radius;
  constants.push_back (decayInfo->motherMass);
  constants.push_back (decayInfo->daug1Mass);
  constants.push_back (decayInfo->daug2Mass);
  constants.push_back (decayInfo->daug3Mass);
  constants.push_back (decayInfo->meson_radius);
  //MEMCPY_TO_SYMBOL(functorConstants, decayConstants, 5*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice);  

  pindices.push_back(decayInfo->resonances.size()); 
  constants.push_back (decayInfo->resonances.size ());
  static int cacheCount = 0; 
  cacheToUse = cacheCount++; 
  pindices.push_back(cacheToUse); 
  constants.push_back (cacheToUse);

  for (std::vector<ResonancePdf*>::iterator res = decayInfo->resonances.begin(); res != decayInfo->resonances.end(); ++res)
  {
    pindices.push_back(registerParameter((*res)->amp_real));
    parameterList.push_back ((*res)->amp_real);

    pindices.push_back(registerParameter((*res)->amp_imag));
    parameterList.push_back ((*res)->amp_imag);

    pindices.push_back((*res)->getFunctionIndex());
    pindices.push_back((*res)->getParameterIndex());
    //(*res)->setConstantIndex(cIndex); 
    components.push_back(*res);
  }

  pindices.push_back(efficiency->getFunctionIndex());
  pindices.push_back(efficiency->getParameterIndex());
  components.push_back(efficiency); 

  GET_FUNCTION_ADDR(ptr_to_DalitzPlot);
  initialise(pindices);

  redoIntegral = new bool[decayInfo->resonances.size()];
  cachedMasses = new fptype[decayInfo->resonances.size()];
  cachedWidths = new fptype[decayInfo->resonances.size()];
  integrals    = new devcomplex<fptype>**[decayInfo->resonances.size()];
  integrators  = new SpecialResonanceIntegrator**[decayInfo->resonances.size()];
  calculators  = new SpecialResonanceCalculator*[decayInfo->resonances.size()];

  for (int i = 0; i < decayInfo->resonances.size(); ++i) {
    redoIntegral[i] = true;
    cachedMasses[i] = -1;
    cachedWidths[i] = -1; 
    integrators[i]  = new SpecialResonanceIntegrator*[decayInfo->resonances.size()];
    calculators[i]  = new SpecialResonanceCalculator(parameters, i); 
    integrals[i]    = new devcomplex<fptype>*[decayInfo->resonances.size()];
    
    for (int j = 0; j < decayInfo->resonances.size(); ++j) {
      integrals[i][j]   = new devcomplex<fptype>(0, 0); 
      integrators[i][j] = new SpecialResonanceIntegrator(parameters, i, j); 
    }
  }

  addSpecialMask(PdfBase::ForceSeparateNorm); 
}

__host__ void DalitzPlotPdf::setDataSize (unsigned int dataSize, unsigned int evtSize) {
  // Default 3 is m12, m13, evtNum
  totalEventSize = evtSize;
  assert(totalEventSize >= 3); 

  if (cachedWaves) delete cachedWaves;

  numEntries = dataSize; 

#ifdef TARGET_MPI
  cachedWaves = new DEVICE_VECTOR<devcomplex<fptype> >(m_iEventsPerTask*decayInfo->resonances.size());
#else
  cachedWaves = new DEVICE_VECTOR<devcomplex<fptype> >(numEntries*decayInfo->resonances.size());
#endif
  void* dummy = thrust::raw_pointer_cast(cachedWaves->data()); 
  MEMCPY_TO_SYMBOL(cResonances, &dummy, sizeof(devcomplex<fptype>*), cacheToUse*sizeof(devcomplex<fptype>*), cudaMemcpyHostToDevice); 
  setForceIntegrals(); 
}


__host__ void DalitzPlotPdf::recursiveSetIndices ()
{
  //(brad): copy into our device list, will need to have a variable to determine type
  GET_FUNCTION_ADDR(ptr_to_Polynomial);
  host_function_table[num_device_functions] = host_fcn_ptr;
  functionIdx = num_device_functions;
  num_device_functions ++;
  
  //(brad): confused by this parameters variable.  Wouldn't each PDF get the current total, not the current amount?
  //parameters = totalParams;
  //totalParams += (2 + pindices.size() + observables.size());

  //in order to figure out the next index, we will need to do some additions to get all the proper offsets
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

  //add efficiency function here.  Or maybe after paremeters...
  for (int i = 0; i < components.size (); i++)
    components[i]->recursiveSetIndices ();

  generateNormRange();

  //add efficiency function here.  Or maybe after paremeters...
  //we need to add the special resonance calculator and integrator here
  GET_FUNCTION_ADDR(device_DalitzPlot_calcIntegrals);
  host_function_table[num_device_functions] = host_fcn_ptr;
  integrateIdx = num_device_functions;
  num_device_functions++;

  //now loop over all the resonances, populate the cudaArray
  for (std::vector<ResonancePdf*>::iterator res = decayInfo->resonances.begin(); res != decayInfo->resonances.end(); ++res)
  {
    (*res)->recursiveSetIndices();
  }

  //we need to pass the resonance functions here.  Unsure how these are determined...
  integrateParams = totalParams;

  GET_FUNCTION_ADDR(device_DalitzPlot_calcIntegrals);
  host_function_table[num_device_functions] = host_fcn_ptr;
  integrateIdx = num_device_functions;
  num_device_functions++;

  calculateParams = totalParams;
}

__host__ fptype DalitzPlotPdf::normalise ()
{
  recursiveSetNormalisation(1); // Not going to normalise efficiency, 
  // so set normalisation factor to 1 so it doesn't get multiplied by zero. 
  // Copy at this time to ensure that the SpecialResonanceCalculators, which need the efficiency, 
  // don't get zeroes through multiplying by the normFactor. 
  //MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  int totalBins = _m12->numbins * _m13->numbins;
  if (!dalitzNormRange) {
    gooMalloc((void**) &dalitzNormRange, 6*sizeof(fptype));
  
    fptype* host_norms = new fptype[6];
    host_norms[0] = _m12->lowerlimit;
    host_norms[1] = _m12->upperlimit;
    host_norms[2] = _m12->numbins;
    host_norms[3] = _m13->lowerlimit;
    host_norms[4] = _m13->upperlimit;
    host_norms[5] = _m13->numbins;
    MEMCPY(dalitzNormRange, host_norms, 6*sizeof(fptype), cudaMemcpyHostToDevice);
    delete[] host_norms; 
  }

  for (unsigned int i = 0; i < decayInfo->resonances.size(); ++i) {
    redoIntegral[i] = forceRedoIntegrals; 
    if (!(decayInfo->resonances[i]->parametersChanged())) continue;
    redoIntegral[i] = true; 
    decayInfo->resonances[i]->storeParameters();
  }
  forceRedoIntegrals = false; 

  // Only do this bit if masses or widths have changed.  
  thrust::constant_iterator<fptype*> arrayAddress(dalitzNormRange); 
  thrust::counting_iterator<int> binIndex(0); 

  // NB, SpecialResonanceCalculator assumes that fit is unbinned! 
  // And it needs to know the total event size, not just observables
  // for this particular PDF component. 
  thrust::constant_iterator<fptype*> dataArray(dev_event_array); 
  thrust::constant_iterator<int> eventSize(totalEventSize);
  thrust::counting_iterator<int> eventIndex(0);
  thrust::constant_iterator<int> funcIdx (calculateIdx);
  thrust::constant_iterator<int> paramIdx (calculateParams);

  for (int i = 0; i < decayInfo->resonances.size(); ++i) {
    if (redoIntegral[i])
    {
#ifdef TARGET_MPI
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, dataArray, eventSize)),
  			thrust::make_zip_iterator(thrust::make_tuple(eventIndex + m_iEventsPerTask, arrayAddress, eventSize)),
			strided_range<DEVICE_VECTOR<devcomplex<fptype> >::iterator>(
				cachedWaves->begin() + i, 
				cachedWaves->end(), 
				decayInfo->resonances.size()).begin(), 
			*(calculators[i]));
#else
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, funcIdx, paramIdx, dataArray, eventSize)),
  			thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, funcIdx, paramIdx, arrayAddress, eventSize)),
			strided_range<DEVICE_VECTOR<devcomplex<fptype> >::iterator>(
				cachedWaves->begin() + i, 
				cachedWaves->end(), 
				decayInfo->resonances.size()).begin(), 
			*(calculators[i]));
#endif
    }
 
    // Possibly this can be done more efficiently by exploiting symmetry? 
    for (int j = 0; j < decayInfo->resonances.size(); ++j) {
      if ((!redoIntegral[i]) && (!redoIntegral[j])) continue; 
      devcomplex<fptype> dummy(0, 0);
      thrust::plus<devcomplex<fptype> > complexSum; 
      thrust::constant_iterator<int> funcIdx (integrateIdx);
      thrust::constant_iterator<int> paramIdx (integrateParams);

      //we need to pass the integration function here
      /*(*(integrals[i][j])) = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(binIndex, arrayAddress)),
						      thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, arrayAddress)),
						      *(integrators[i][j]), 
						      dummy, 
						      complexSum);*/

      (*(integrals[i][j])) = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(binIndex, funcIdx, paramIdx, arrayAddress)),
						      thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, funcIdx, paramIdx, arrayAddress)),
						      *(integrators[i][j]), dummy, complexSum);
    }
  } 

  // End of time-consuming integrals. 
  complex<fptype> sumIntegral(0, 0);
  for (unsigned int i = 0; i < decayInfo->resonances.size(); ++i) {
    //int param_i = parameters + resonanceOffset_DP + resonanceSize*i; 
    complex<fptype> amplitude_i(host_params[constantsIdx + 4 + i*resonanceSize], host_params[constantsIdx + 4 + i*resonanceSize]);
    for (unsigned int j = 0; j < decayInfo->resonances.size(); ++j) {
      //int param_j = parameters + resonanceOffset_DP + resonanceSize*j; 
      complex<fptype> amplitude_j(host_params[constantsIdx + 4 + j*resonanceSize], -host_params[constantsIdx + 4 + j*resonanceSize]); 
      // Notice complex conjugation

      sumIntegral += (amplitude_i * amplitude_j * complex<fptype>((*(integrals[i][j])).real, (*(integrals[i][j])).imag)); 
    }
  }

  fptype ret = real(sumIntegral); // That complex number is a square, so it's fully real
  double binSizeFactor = 1;
  binSizeFactor *= ((_m12->upperlimit - _m12->lowerlimit) / _m12->numbins);
  binSizeFactor *= ((_m13->upperlimit - _m13->lowerlimit) / _m13->numbins);
  ret *= binSizeFactor;

  host_params[normalisationIdx] = 1.0/ret;
  return (fptype) ret; 
}

SpecialResonanceIntegrator::SpecialResonanceIntegrator (int pIdx, unsigned int ri, unsigned int rj) 
  : resonance_i(ri)
  , resonance_j(rj)
  , parameters(pIdx) 
{}

EXEC_TARGET devcomplex<fptype> SpecialResonanceIntegrator::operator () (thrust::tuple<int, int, int, fptype*> t) const
{
  // (brad): New indexing:  bin number, funcIdx, paramIdx, fptype with the actual bin numbers
  

  // Bin index, base address [lower, upper, numbins] 
  // Notice that this is basically MetricTaker::operator (binned) with the special-case knowledge
  // that event size is two, and that the function to call is dev_DalitzPlot_calcIntegrals.

  int globalBinNumber  = thrust::get<0>(t);
  unsigned int funcIdx          = thrust::get<1>(t);
  unsigned int paramIdx         = thrust::get<2>(t);

  fptype lowerBoundM12 = thrust::get<3>(t)[0];
  fptype upperBoundM12 = thrust::get<3>(t)[1];  
  int numBinsM12       = (int) FLOOR(thrust::get<3>(t)[2] + 0.5); 
  int binNumberM12     = globalBinNumber % numBinsM12;
  fptype binCenterM12  = upperBoundM12 - lowerBoundM12;
  binCenterM12        /= numBinsM12;
  binCenterM12        *= (binNumberM12 + 0.5); 
  binCenterM12        += lowerBoundM12; 

  globalBinNumber     /= numBinsM12; 
  fptype lowerBoundM13 = thrust::get<3>(t)[3];
  fptype upperBoundM13 = thrust::get<3>(t)[4];  
  int numBinsM13       = (int) FLOOR(thrust::get<3>(t)[5] + 0.5); 
  fptype binCenterM13  = upperBoundM13 - lowerBoundM13;
  binCenterM13        /= numBinsM13;
  binCenterM13        *= (globalBinNumber + 0.5); 
  binCenterM13        += lowerBoundM13; 

  unsigned int* indices = NULL;   
  devcomplex<fptype> ret = device_DalitzPlot_calcIntegrals(binCenterM12, binCenterM13, resonance_i, resonance_j, &funcIdx, &paramIdx); 

  fptype fakeEvt[10]; // Need room for many observables in case m12 or m13 were assigned a high index in an event-weighted fit. 
  fakeEvt[indices[indices[0] + 2 + 0]] = binCenterM12;
  fakeEvt[indices[indices[0] + 2 + 1]] = binCenterM13;
  unsigned int numResonances = indices[2]; 
  unsigned int effFunctionIdx = parIndexFromResIndex_DP(numResonances); 

  fptype eff = callFunction(fakeEvt, &effFunctionIdx, indices); 

  // Multiplication by eff, not sqrt(eff), is correct:
  // These complex numbers will not be squared when they
  // go into the integrals. They've been squared already,
  // as it were. 
  ret *= eff;
  return ret; 
}

SpecialResonanceCalculator::SpecialResonanceCalculator (int pIdx, unsigned int res_idx) 
  : resonance_i(res_idx)
  , parameters(pIdx)
{}

EXEC_TARGET devcomplex<fptype> SpecialResonanceCalculator::operator () (thrust::tuple<int, int, int, fptype*, int> t) const {
  // Calculates the BW values for a specific resonance. 
  devcomplex<fptype> ret;
  int evtNum = thrust::get<0>(t); 
  fptype* evt = thrust::get<3>(t) + (evtNum * thrust::get<2>(t)); 

  unsigned int funcIdx = thrust::get<1>(t);
  unsigned int paramIdx = thrust::get<2>(t);

  unsigned int* indices = NULL;   // Jump to DALITZPLOT position within parameters array
  fptype m12 = evt[indices[2 + indices[0]]]; 
  fptype m13 = evt[indices[3 + indices[0]]];

  fptype motherMass = cudaArray[indices[1] + 0]; 
  fptype daug1Mass  = cudaArray[indices[1] + 1]; 
  fptype daug2Mass  = cudaArray[indices[1] + 2]; 
  fptype daug3Mass  = cudaArray[indices[1] + 3];  
  if (!inDalitz(m12, m13, motherMass, daug1Mass, daug2Mass, daug3Mass)) return ret;
  fptype m23 = motherMass*motherMass + daug1Mass*daug1Mass + daug2Mass*daug2Mass + daug3Mass*daug3Mass - m12 - m13; 

  //int parameter_i = parIndexFromResIndex_DP(resonance_i); // Find position of this resonance relative to DALITZPLOT start 

  //unsigned int functn_i = indices[parameter_i+2];
  //unsigned int params_i = indices[parameter_i+3];

  ret = getResonanceAmplitude(m12, m13, m23, &funcIdx, &paramIdx);
  //printf("Amplitude %f %f %f (%f, %f)\n ", m12, m13, m23, ret.real, ret.imag); 
  return ret;
}

