#include "PolynomialPdf.hh"

EXEC_TARGET fptype device_Polynomial (fptype* evt, unsigned int *funcIdx, unsigned int* indices) {
  // Structure is nP lowestdegree c1 c2 c3 nO o1

  int numParams = cudaArray[*indices + 0]; 
  int numObs = cudaArray[*indices + numParams + 1];
  int numCons = cudaArray[*indices + numParams + 1 + numObs + 1];
  int lowestDegree = cudaArray[*indices + numParams + 1 + numObs + 1 + 1]; 

  fptype x = evt[0];//indices[2 + indices[0]]]; 
  fptype ret = 0; 
  for (int i = 1; i < numParams; ++i)
  {
    ret += cudaArray[*indices + i] * POW(x, lowestDegree + i - 1); 
  }

  *indices += 7;
  *funcIdx += 1;

  return ret; 
}

EXEC_TARGET fptype device_OffsetPolynomial (fptype* evt, unsigned int *funcIdx, unsigned int* indices) {
  //int numParams = indices[0]; 
  //int lowestDegree = indices[1]; 

  int numParams = cudaArray[*indices + 0];
  int numObs = cudaArray[*indices + numParams + 1];
  int numCons = cudaArray[*indices + numParams + 1 + numObs + 1];
  int lowestDegree = cudaArray[*indices + numParams + 1 + numObs + 1 + 1];

  //fptype x = evt[indices[2 + numParams]];
  fptype x = evt[0];
  //todo: (brad) I think this is pointing to an observable?
  //x -= cudaArray[*indices + numParams]; 
  x -= numParams;

  fptype ret = 0; 
  for (int i = 2; i < numParams; ++i) {
    ret += cudaArray[*indices + i] * POW(x, lowestDegree + i - 2); 
  }

  *indices += 7;
  *funcIdx += 1;

  return ret; 
}

EXEC_TARGET fptype device_MultiPolynomial (fptype* evt, unsigned int* funcIdx, unsigned int* indices) {
  // Structure is nP, maxDegree, offset1, offset2, ..., coeff1, coeff2, ..., nO, o1, o2, ... 
  //int idx[2];
  //idx[0] = indices[0];
  //idx[1] = indices[1];

  //int numObservables = indices[idx[0] + 1]; 
  //int maxDegree = idx[1] + 1; 

  int numParams = cudaArray[*indices + 0];
  int numObs = cudaArray[*indices + numParams + 1];
  int numCons = cudaArray[*indices + numParams + 1 + numObs + 1];
  int maxDegree = cudaArray[*indices + numParams + 1 + numObs + 1 + 1] + 1;

  // Only appears in construction (maxDegree + 1) or (x > maxDegree), so
  // may as well add the one and use >= instead. 

  // Technique is to iterate over the full n-dimensional box, skipping matrix elements
  // whose sum of indices is greater than maxDegree. Notice that this is increasingly
  // inefficient as n grows, since a larger proportion of boxes will be skipped. 
  int numBoxes = 1;
  for (int i = 0; i < numObs; ++i)
    numBoxes *= maxDegree; 

  int coeffNumber = 2; // Index of first coefficient is 2 + nO, not 1 + nO, due to maxDegree. (nO comes from offsets.) 
  fptype ret = cudaArray[*indices + numParams + 1 + numObs + 1 + coeffNumber++]; // Coefficient of constant term. 
  for (int i = 1; i < numBoxes; ++i) { // Notice skip of inmost 'box' in the pyramid, corresponding to all powers zero, already accounted for. 
    fptype currTerm = 1; 
    int currIndex = i; 
    int sumOfIndices = 0; 
    //if ((gpuDebug & 1) && (THREADIDX == 50) && (BLOCKIDX == 3))
    //if ((BLOCKIDX == internalDebug1) && (THREADIDX == internalDebug2)) 
    //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
    //printf("[%i, %i] Start box %i %f %f:\n", BLOCKIDX, THREADIDX, i, ret, evt[8]);
    for (int j = 0; j < numObs; ++j) {
      //int tmp[2];
      //tmp[0] = indices[2 + j];
      //tmp[1] = indices[2 + idx[0] + j];

      //todo (brad): need to debug these, what is offset and x supposed to be?
      fptype offset = cudaArray[*indices + numParams + 1 + i]; // x0, y0, z0... 
      //todo (brad): need to debug these, what is offset and x supposed to be?
      fptype x = evt[0]; // x, y, z...    

      x -= offset; 
      int currPower = currIndex % maxDegree; 
      currIndex /= maxDegree; 
      currTerm *= POW(x, currPower);
      sumOfIndices += currPower; 
      //if ((gpuDebug & 1) && (THREADIDX == 50) && (BLOCKIDX == 3))
      //if ((BLOCKIDX == internalDebug1) && (THREADIDX == internalDebug2)) 
      //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
      //printf("  [%f -> %f^%i = %f] (%i %i) \n", evt[indices[2 + indices[0] + j]], x, currPower, POW(x, currPower), sumOfIndices, indices[2 + indices[0] + j]); 
    }
    //if ((gpuDebug & 1) && (THREADIDX == 50) && (BLOCKIDX == 3))
    //if ((BLOCKIDX == internalDebug1) && (THREADIDX == internalDebug2)) 
    //printf(") End box %i\n", i);
    // All threads should hit this at the same time and with the same result. No branching. 
    if (sumOfIndices >= maxDegree) continue; 
    fptype coefficient = cudaArray[*indices + numParams + 1 + numObs + 1 + coeffNumber++]; // Coefficient from MINUIT
    //if ((gpuDebug & 1) && (THREADIDX == 50) && (BLOCKIDX == 3))
    //if ((BLOCKIDX == internalDebug1) && (THREADIDX == internalDebug2)) 
    //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
    //printf("Box %i contributes %f * %f = %f -> %f\n", i, currTerm, p[indices[coeffNumber - 1]], coefficient*currTerm, (ret + coefficient*currTerm)); 
    currTerm *= coefficient;
    ret += currTerm; 
  }

  //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
  //printf("Final polynomial: %f\n", ret); 

  *indices += numParams + numObs + numCons + 2;
  *funcIdx += 1;

  //if (0 > ret) ret = 0; 
  return ret; 
}


MEM_DEVICE device_function_ptr ptr_to_Polynomial = device_Polynomial; 
MEM_DEVICE device_function_ptr ptr_to_OffsetPolynomial = device_OffsetPolynomial; 
MEM_DEVICE device_function_ptr ptr_to_MultiPolynomial = device_MultiPolynomial; 

// Constructor for single-variate polynomial, with optional zero point. 
__host__ PolynomialPdf::PolynomialPdf (string n, Variable* _x, vector<Variable*> weights, Variable* x0, unsigned int lowestDegree) 
  : GooPdf(_x, n) 
  , center(x0)
{

  vector<unsigned int> pindices;
  pindices.push_back(lowestDegree);
  constants.push_back (lowestDegree);
  for (vector<Variable*>::iterator v = weights.begin(); v != weights.end(); ++v) {
    pindices.push_back(registerParameter(*v));
    //parameterList.push_back (*v);
  } 
  if (x0) {
    polyType = 1;
    pindices.push_back(registerParameter(x0));
    //parameterList.push_back (x0);
    GET_FUNCTION_ADDR(ptr_to_OffsetPolynomial);
  }
  else {
    polyType = 0;
    GET_FUNCTION_ADDR(ptr_to_Polynomial);
  }
  initialise(pindices); 
}

// Constructor for multivariate polynomial. 
__host__ PolynomialPdf::PolynomialPdf (string n, vector<Variable*> obses, vector<Variable*> coeffs, vector<Variable*> offsets, unsigned int maxDegree) 
  : GooPdf(0, n) 
{
  unsigned int numParameters = 1; 
  // For 1 observable, equal to n = maxDegree + 1. 
  // For two, n*(n+1)/2, ie triangular number. This generalises:
  // 3: Pyramidal number n*(n+1)*(n+2)/(3*2)
  // 4: Hyperpyramidal number n*(n+1)*(n+2)*(n+3)/(4*3*2)
  // ...
  for (unsigned int i = 0; i < obses.size(); ++i) {
    registerObservable(obses[i]);
    numParameters *= (maxDegree + 1 + i); 
    observables.push_back (obses[i]);
  }
  for (int i = observables.size(); i > 1; --i) numParameters /= i; 

  while (numParameters > coeffs.size()) {
    char varName[100]; 
    sprintf(varName, "%s_extra_coeff_%i", getName().c_str(), (int) coeffs.size());
    
    Variable* newTerm = new Variable(varName, 0);
    coeffs.push_back(newTerm); 
    
    cout << "Warning: " << getName() << " created dummy variable "
	      << varName
	      << " (fixed at zero) to account for all terms.\n";
  }

  while (offsets.size() < obses.size()) {
    char varName[100]; 
    sprintf(varName, "%s_extra_offset_%i", getName().c_str(), (int) offsets.size());
    Variable* newOffset = new Variable(varName, 0); 
    offsets.push_back(newOffset); 
  }

  vector<unsigned int> pindices;
  pindices.push_back(maxDegree);
  constants.push_back (maxDegree);
  for (vector<Variable*>::iterator o = offsets.begin(); o != offsets.end(); ++o) {
    pindices.push_back(registerParameter(*o)); 
    //parameterList.push_back (*o);
  }
  for (vector<Variable*>::iterator c = coeffs.begin(); c != coeffs.end(); ++c) {
    pindices.push_back(registerParameter(*c));
    //parameterList.push_back (*c);
  }

  polyType = 2;
  GET_FUNCTION_ADDR(ptr_to_MultiPolynomial);
  initialise(pindices); 
}

PolynomialPdf::~PolynomialPdf ()
{
}

__host__ void PolynomialPdf::recursiveSetIndices ()
{
  //(brad): copy into our device list, will need to have a variable to determine type
  if (polyType == 0)
    GET_FUNCTION_ADDR(ptr_to_Polynomial);
  else if (polyType == 1)
    GET_FUNCTION_ADDR(ptr_to_OffsetPolynomial);
  else if (polyType == 2)
    GET_FUNCTION_ADDR(ptr_to_MultiPolynomial);
  
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

  for (int i = 0; i < components.size (); i++)
    components[i]->recursiveSetIndices ();

  generateNormRange();
}

__host__ fptype PolynomialPdf::integrate (fptype lo, fptype hi) {
  // This is *still* wrong. (13 Feb 2013.) 
  unsigned int* indices = host_indices+parameters; 
  fptype lowestDegree = indices[1]; 

  if (center) {
    hi -= host_params[indices[indices[0]]];
    lo -= host_params[indices[indices[0]]];
  }

  fptype ret = 0; 
  for (int i = 2; i < indices[0] + (center ? 0 : 1); ++i) {
    fptype powerPlusOne = lowestDegree + i - 2; 
    fptype curr = POW(hi, powerPlusOne); 
    curr       -= POW(lo, powerPlusOne); 
    curr       /= powerPlusOne; 
    ret        += host_params[indices[i]] * curr; 
  }

  return ret; 
}

__host__ fptype PolynomialPdf::getCoefficient (int coef) {
  // NB! This function only works for single polynomials. 
  if (1 != observables.size()) {
    std::cout << "Warning: getCoefficient method of PolynomialPdf not implemented for multi-dimensional polynomials. Returning zero, which is very likely wrong.\n"; 
    return 0; 
  }

  unsigned int* indices = host_indices + parameters;

  // True function is, say, ax^2 + bx + c.
  // We express this as (a'x^2 + b'x + c')*N.
  // So to get the true coefficient, multiply the internal
  // one by the normalisation. (In non-PDF cases the normalisation
  // equals one, which gives the same result.)

  // Structure is nP lowestdegree c1 c2 c3 nO o1
  if (coef < indices[1]) return 0; // Less than least power. 
  if (coef > indices[1] + (indices[0] - 1)) return 0; // Greater than max power. 

  fptype norm = normalise(); 
  norm = (1.0 / norm); 

  fptype param = host_params[indices[2 + coef - indices[1]]]; 
  return norm*param; 
}
