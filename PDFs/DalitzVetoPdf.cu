#include "DalitzVetoPdf.hh"
#include "DalitzPlotHelpers.hh" 

EXEC_TARGET fptype device_DalitzVeto (unsigned int eventId, unsigned int *funcIdx, unsigned int* indices) {
  int numParams = cudaArray[*indices];
  fptype x         = cudaArray[*indices + 1]; 
  fptype y         = cudaArray[*indices + 2]; 

  fptype motherM   = cudaArray[*indices + 3];
  fptype d1m       = cudaArray[*indices + 4];
  fptype d2m       = cudaArray[*indices + 5];
  fptype d3m       = cudaArray[*indices + 6];

  fptype motherM2  = motherM*motherM;
  fptype d1m2      = d1m*d1m;
  fptype d2m2      = d2m*d2m;
  fptype d3m2      = d3m*d3m;

  fptype massSum   = motherM2 + d1m2 + d2m2 + d3m2;

  fptype ret = inDalitz(x, y, motherM, d1m, d2m, d3m) ? 1.0 : 0.0; 
  int numConstants = cudaArray[*indices + 2 + numParams];
  int consIdx = numParams + 2;
  unsigned int numVetos = cudaArray[*indices + consIdx];

  fptype z         = massSum - x - y;

  for (int i = 0; i < numVetos; ++i) {
    int i2 = i*2;

    unsigned int varIndex = cudaArray[*indices + consIdx + 1 + i];
    fptype minimum        = cudaArray[*indices + 7 + i2 + 1];
    fptype maximum        = cudaArray[*indices + 7 + i2 + 2];
    fptype currDalitzVar = (PAIR_12 == varIndex ? x : PAIR_13 == varIndex ? y : z);

    ret *= ((currDalitzVar < maximum) && (currDalitzVar > minimum)) ? 0.0 : 1.0;
  }

  *funcIdx += 1;
  *indices += numParams + 1 + 2 + numConstants + 1 + 2;

  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_DalitzVeto = device_DalitzVeto;

__host__ DalitzVetoPdf::DalitzVetoPdf (std::string n, Variable* _x, Variable* _y, Variable* motherM, Variable* d1m, Variable* d2m, Variable* d3m, vector<VetoInfo*> vetos) 
  : GooPdf(0, n) 
{
  registerObservable(_x);
  registerObservable(_y);
  parameterList.push_back (_x);
  parameterList.push_back (_y);

  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(motherM));
  pindices.push_back(registerParameter(d1m));
  pindices.push_back(registerParameter(d2m));
  pindices.push_back(registerParameter(d3m));
  parameterList.push_back (motherM);
  parameterList.push_back (d1m);
  parameterList.push_back (d2m);
  parameterList.push_back (d3m);

  pindices.push_back(vetos.size()); 
  constants.push_back (vetos.size ());
  for (vector<VetoInfo*>::iterator v = vetos.begin(); v != vetos.end(); ++v) {
    pindices.push_back((*v)->cyclic_index);
    pindices.push_back(registerParameter((*v)->minimum));
    pindices.push_back(registerParameter((*v)->maximum));

    constants.push_back ((*v)->cyclic_index);
    parameterList.push_back ((*v)->minimum);
    parameterList.push_back ((*v)->maximum);
  }

  GET_FUNCTION_ADDR(ptr_to_DalitzVeto);
  initialise(pindices); 
}

__host__ void DalitzVetoPdf::recursiveSetIndices ()
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

  for (int i = 0; i < components.size (); i++)
    components[i]->recursiveSetIndices ();

  generateNormRange();
}
