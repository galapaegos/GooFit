#include "GaussianPdf.hh"

EXEC_TARGET fptype device_Gaussian (fptype* evt, unsigned int* funcIdx, unsigned int* indices)
{
  //int tidx = blockDim.x *blockIdx.x + threadIdx.x;

  fptype x = evt[0];//indices[2 + indices[0]]]; 

  //__shared__ fptype idx[2];
  //if (THREADIDX == 0)
  //{
  //  idx[0] = cudaArray[*indices + 1];
  //  idx[1] = cudaArray[*indices + 2];
  //}

   //__syncthreads();
  
  fptype mean = cudaArray[*indices + 1];
  fptype sigma = cudaArray[*indices + 2];

  //fptype ret = EXP(-0.5*(x-idx[0])*(x-idx[0])/(idx[1]*idx[1]));
  fptype ret = EXP(-0.5*(x-mean)*(x-mean)/(sigma*sigma));

  *indices += 8;
  *funcIdx += 1;

  //if ((0 == THREADIDX) && (0 == BLOCKIDX)) cuPrintf("Gaussian Values %f %i %i %f %f %i\n", x, indices[1], indices[2], mean, sigma, callnumber); 
  //cuPrintf("device_Gaussian %f %i %i %f %f %i %p %f\n", x, indices[1], indices[2], mean, sigma, callnumber, indices, ret); 
  //if ((0 == THREADIDX) && (0 == BLOCKIDX))
  //printf("device_Gaussian %f %f %f %i %f\n", x, mean, sigma, callnumber, ret);     


  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_Gaussian = device_Gaussian; 

__host__ GaussianPdf::GaussianPdf (std::string n, Variable* _x, Variable* mean, Variable* sigma) 
  : GooPdf(_x, n) 
{
  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(mean));
  pindices.push_back(registerParameter(sigma));
  GET_FUNCTION_ADDR(ptr_to_Gaussian);
  initialise(pindices); 

  m_pSigma = sigma;
}

GaussianPdf::~GaussianPdf ()
{
}

__host__ void GaussianPdf::recursiveSetIndices()
{
  //(brad): all pdfs need to add themselves to device list so we can increment
  GET_FUNCTION_ADDR(ptr_to_Gaussian);
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
    components[i]->recursiveSetIndices();

  generateNormRange ();
}

__host__ fptype GaussianPdf::integrate (fptype lo, fptype hi) {
  //static const fptype root2 = sqrt(2.);
  static const fptype rootPi = sqrt(atan2(0.0,-1.0));
  //static const fptype rootPiBy2 = rootPi / root2;

  //unsigned int* indices = host_indices+parameters; 
  //fptype xscale = root2*host_params[indices[2]];

  /*
  std::cout << "Gaussian integral: " 
	    << xscale << " "
	    << host_params[indices[1]] << " "
	    << host_params[indices[2]] << " "
	    << ERF((hi-host_params[indices[1]])/xscale) << " "
	    << ERF((lo-host_params[indices[1]])/xscale) << " "
	    << rootPiBy2*host_params[indices[2]]*(ERF((hi-host_params[indices[1]])/xscale) -
						  ERF((lo-host_params[indices[1]])/xscale)) 
	    << std::endl; 
  */
  //return rootPiBy2*host_params[indices[2]]*(ERF((hi-host_params[indices[1]])/xscale) - 
  //					    ERF((lo-host_params[indices[1]])/xscale));

  // Integral over all R. 

  //we aren't populating host_params, so grab the parameter from the vec
  fptype sigma = m_pSigma->value + m_pSigma->blind;

  //fptype sigma = host_params[indices[2]];
  sigma *= root2*rootPi;
  return sigma; 
}

