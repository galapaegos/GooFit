#include "PdfBase.hh"

// This is code that belongs to the PdfBase class, that is, 
// it is common across all implementations. But it calls on device-side
// functions, and due to the nvcc translation-unit limitations, it cannot
// sit in its own object file; it must go in the CUDAglob.cu. So it's
// off on its own in this inline-cuda file, which GooPdf.cu 
// should include. 

#ifdef CUDAPRINT
__host__ void PdfBase::copyParams (const std::vector<double>& pars) const {
  if (host_callnumber < 1) {
    std::cout << "Copying parameters: " << (long long) cudaArray << " ";
  }
  for (unsigned int i = 0; i < pars.size(); ++i) {
    host_params[parametersIdx + i] = pars[i]; 
    
    if (host_callnumber < 1) {
      std::cout << pars[i] << " ";
    }
    
    if (isnan(host_params[i])) {
      std::cout << " agh, NaN, die " << i << std::endl;
      abortWithCudaPrintFlush(__FILE__, __LINE__, "NaN in parameter"); 
    }
  }
  
  if (host_callnumber < 1) {
    std::cout << std::endl; 
  }
  MEMCPY_TO_SYMBOL(cudaArray, host_params, pars.size()*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
}
#else 
__host__ void PdfBase::copyParams (const std::vector<double>& pars) const {
  // copyParams method performs eponymous action! 

  for (unsigned int i = 0; i < pars.size(); ++i)
  {
    host_params[parametersIdx + i] = pars[i];
    
    if (std::isnan(host_params[i])) {
      std::cout << " agh, parameter is NaN, die " << i << std::endl;
      abortWithCudaPrintFlush(__FILE__, __LINE__, "NaN in parameter"); 
    }
  }

  MEMCPY_TO_SYMBOL(cudaArray, host_params, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
}
#endif

__host__ void PdfBase::copy (std::vector<Variable*> vars)
{
  copyParams (vars);
  MEMCPY_TO_SYMBOL(cudaArray, host_params, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
}

__host__ void PdfBase::copyParams (std::vector<Variable*> vars) {
  //(brad) copy all values into host_params to be transfered
  //note these are indexed the way they are passed.
  // Copies values of Variable objects

  //copy from vars to our local copy, should be able to remove this at some point(?)
  for (int x = 0; x < vars.size (); x++)
  {
    for (int y = 0; y < parameterList.size (); y++)
    {
      if (parameterList[y]->name == vars[x]->name)
      {
        parameterList[y]->value = vars[x]->value;
        parameterList[y]->blind = vars[x]->blind;
      }
    }
  }

  for (int i = 0; i < parameterList.size (); i++)
  {
    host_params[parametersIdx + i] = parameterList[i]->value + parameterList[i]->blind;
    //printf ("param[%i]:%f\n", i,parameterList[i]->value + parameterList[i]->blind); 
  }

  //recurse
  for (int i = 0; i < components.size (); i++)
    components[i]->copyParams(vars);

  //parCont pars; 
  //getParameters(pars); 
  //std::vector<double> values; 
  //for (parIter v = pars.begin(); v != pars.end(); ++v) {
  //  int index = (*v)->getIndex(); 
  //  if (index >= (int) values.size()) values.resize(index + 1);
  //  values[index] = (*v)->value;
  //}
  //copyParams(values); 
}

//__host__ void PdfBase::copyNormFactors () const {
  //MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
//  SYNCH(); // Ensure normalisation integrals are finished
//}

__host__ void PdfBase::initialiseIndices (std::vector<unsigned int> pindices) {
  // Structure of the individual index array: Number of parameters, then the indices
  // requested by the subclass (which will be interpreted by the subclass kernel), 
  // then the number of observables, then the observable indices. Notice that the
  // observable indices are not set until 'setIndices' is called, usually from setData;
  // here we only reserve space for them by setting totalParams. 
  // This is to allow index sharing between PDFs - all the PDFs must be constructed 
  // before we know what observables exist. 

  //(brad) note we are changing how these are storing:
  if (totalParams + pindices.size() >= maxParams) {
    std::cout << "Major problem with pindices size: " << totalParams << " + " << pindices.size() << " >= " << maxParams << std::endl; 
  }

  //(brad): confused by this parameters variable.  Wouldn't each PDF get the current total, not the current amount?
  //parameters = totalParams;
  //totalParams += (2 + pindices.size() + observables.size()); 

  //(brad): First we must pad out how many indices we need.

  //in order to figure out the next index, we will need to do some additions to get all the proper offsets
  host_params[totalParams++] = parameterList.size ();
  parametersIdx = totalParams;
  for (int i = 0; i < pindices.size (); i++)
    host_params[totalParams++] = 0;

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

  std::cout << "pindices: " << pindices.size () << " + observables: " << observables.size () << std::endl;
  std::cout << "host_params after " << getName() << " initialisation: (" << totalParams << ") ";
  for (int i = parametersIdx - 1; i < totalParams; ++i) {
    std::cout << host_params[i] << " ";
  }

  std::cout << std::endl;
  
  std::cout << " | " 
	    << "parameters: " << host_params[parametersIdx - 1] << " " 
	    << "observables: " << host_params[observablesIdx - 1] << " " 
	    << "constants: " << host_params[constantsIdx - 1] << " " 
	    << "normalisation: " << host_params[normalisationIdx - 1] << " " 
	    << std::endl; 

  MEMCPY_TO_SYMBOL(cudaArray, host_params, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
}

__host__ void PdfBase::recursiveSetIndices ()
{
  //special cases will need to overload this function (AddPdf)

  //(brad): Each call needs to set the indices into the array.  In otherwords, each function needs to 'pre-load' 
  //all parameters.  This means that each function will need to read all their variables/parameters they will use
  //before calling any other functions.
  /*
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
  host_params[totalParams++] = 1 + observables.size()*3;
  normalisationIdx = totalParams;
  host_params[totalParams++] = 0;

  for (int i = 0; i < observables.size ()*3; i += 3)
  {
    host_params[totalParams++] = 0;
    host_params[totalParams++] = 0;
    host_params[totalParams++] = 0;
  }
  */

  //(brad): this is overloaded such that each PDF will have its own implementation of recursiveSetIndices
  for (unsigned int i = 0; i < components.size(); ++i)
    components[i]->recursiveSetIndices(); 

  generateNormRange ();
}

__host__ void PdfBase::setIndices () {
  int counter = 0; 
  for (obsIter v = obsBegin(); v != obsEnd(); ++v) {
    (*v)->index = counter++; 
  }

  int test = totalParams;

  //recalculate our indices here, reset:
  totalParams = 0;
  num_device_functions = 0;

  recursiveSetIndices(); 

  MEMCPY_TO_SYMBOL(device_function_table, host_function_table, num_device_functions*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  MEMCPY_TO_SYMBOL(cudaArray, host_params, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
}

__host__ void PdfBase::setData (UnbinnedDataSet* data)
{
  //std::cout << "PdfBase::setData" << std::endl;
  if (dev_event_array) {
    gooFree(dev_event_array);
    SYNCH();
    dev_event_array = 0; 

    m_iEventsPerTask = 0;
  }

  setIndices();
  int dimensions = observables.size();
  numEntries = data->getNumEvents(); 
  numEvents = numEntries; 

#ifdef TARGET_MPI
  int myId, numProcs;
  MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myId);

  int perTask = numEvents/numProcs;

  int *counts = new int[numProcs];
  int *displacements = new int[numProcs];

  //indexing for copying events over!
  for (int i = 0; i < numProcs - 1; i++)
    counts[i] = perTask;
  counts[numProcs - 1] = numEntries - perTask*(numProcs - 1);

  //displacements into the array for indexing!
  displacements[0] = 0;
  for (int i = 1; i < numProcs; i++)
    displacements[i] = displacements[i - 1] + counts[i - 1];
#endif

  //fptype *host_array = new fptype[numEntries*dimensions];
  fptype *host_array = (fptype*)malloc (numEntries*dimensions*sizeof(fptype));

#ifdef TARGET_MPI
  //This is an array to track if we need to redo indexing
  int fixme[observables.size ()];
  memset(fixme, 0, sizeof (int)*observables.size ());

  //printf ("Checking observables for Counts!\n");
  for (int i = 0; i < observables.size (); i++)
  {
    //printf ("%i - %s\n", i, observables[i]->name.c_str ());
    //cast this variable to see if its one we need to correct for
    CountVariable *c = dynamic_cast <CountVariable*> (observables[i]);
    //if it cast, mark it
    if (c)
    {
      fixme[i] = 1;
      //printf ("%i of %i - %s\n", i, observables.size (), c->name.c_str ());
    }
  }
#endif

  //populate this array with our stuff
  for (int i = 0; i < numEntries; ++i)
  {
    //brad: hack I think?  *v->index is not being generated correctly...
    int counter = 0;
    for (obsIter v = obsBegin(); v != obsEnd(); ++v)
    {
      fptype currVal = data->getValue((*v), i);
      host_array[i*dimensions + counter] = currVal; 
      counter++;
    }
  }

#ifdef TARGET_MPI
  // we need to fix our observables indexing to reflect having multiple cards
  for (int i = 1; i < numProcs; i++)
  {
    for (int j = 0; j < counts[i]; j++)
    {
      //assumption is that the last observable is the index!
      for (int k = 0; k < dimensions; k++)
      {
        //Its counting, fix the indexing here
        if (fixme[k] > 0)
          host_array[(j + displacements[i])*dimensions + k] = float (j);
      }
    }
  }
  
  int mystart = displacements[myId];
  int myend = mystart + counts[myId];
  int mycount = myend - mystart;
#endif

#ifdef TARGET_MPI
  gooMalloc((void**) &dev_event_array, dimensions*mycount*sizeof(fptype));
  MEMCPY(dev_event_array, host_array + mystart*dimensions, dimensions*mycount*sizeof(fptype), cudaMemcpyHostToDevice);
  //MEMCPY_TO_SYMBOL(functorConstants, &numEvents, sizeof(fptype), 0, cudaMemcpyHostToDevice);
  delete[] host_array; 

  //update everybody
  setNumPerTask(this, mycount);

  delete [] counts;
  delete [] displacements;
#else
  gooMalloc((void**) &dev_event_array, dimensions*numEntries*sizeof(fptype));
  printf ("dimensions:%i numEntries:%i\n", dimensions, numEntries);
  MEMCPY(dev_event_array, host_array, dimensions*numEntries*sizeof(fptype), cudaMemcpyHostToDevice);
  //delete[] host_array; 
  //free (host_array);
#endif
}

__host__ void PdfBase::setData (BinnedDataSet* data)
{ 
  if (dev_event_array) { 
    gooFree(dev_event_array);
    dev_event_array = 0; 

    m_iEventsPerTask = 0;
  }

  setIndices();
  numEvents = 0; 
  numEntries = data->getNumBins(); 
  int dimensions = 2 + observables.size(); // Bin center (x,y, ...), bin value, and bin volume. 
  if (!fitControl->binnedFit()) setFitControl(new BinnedNllFit()); 

  fptype* host_array = new fptype[numEntries*dimensions]; 

#ifdef TARGET_MPI
  int myId, numProcs;
  MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myId);

  //This is an array to track if we need to redo indexing
  int fixme[dimensions];
  memset(fixme, 0, sizeof (int)*dimensions);

  for (int i = 0; i < observables.size (); i++)
  {
    //cast this variable to see if its one we need to correct for
    CountVariable *c = dynamic_cast <CountVariable*> (observables[i]);
    //if it cast, mark it
    if (c)
      fixme[i] = 1;
  }
#endif

  for (unsigned int i = 0; i < numEntries; ++i) {
    for (obsIter v = obsBegin(); v != obsEnd(); ++v) {
      host_array[i*dimensions + (*v)->index] = data->getBinCenter((*v), i); 
    }

    host_array[i*dimensions + observables.size() + 0] = data->getBinContent(i);
    host_array[i*dimensions + observables.size() + 1] = fitControl->binErrors() ? data->getBinError(i) : data->getBinVolume(i); 
    numEvents += data->getBinContent(i);
  }

#if TARGET_MPI
  int perTask = numEvents/numProcs;

  int *counts = new int[numProcs];
  int *displacements = new int[numProcs];

  //indexing for copying events over!
  for (int i = 0; i < numProcs - 1; i++)
    counts[i] = perTask;
  counts[numProcs - 1] = numEvents - perTask*(numProcs - 1);

  //displacements into the array for indexing!
  displacements[0] = 0;
  for (int i = 1; i < numProcs; i++)
    displacements[i] = displacements[i - 1] + counts[i - 1];

  // we need to fix our observables indexing to reflect having multiple cards
  for (int i = 1; i < numProcs; i++)
  {
    for (int j = 0; j < counts[i]; j++)
    {
      //assumption is that the last observable is the index!
      for (int k = 0; k < dimensions; k++)
      {
        //Its counting, fix the indexing here
        if (fixme[k] > 0)
          host_array[(j + displacements[i])*dimensions + dimensions - k] = float (j);
      }
    }
  }

  int mystart = displacements[myId];
  int myend = mystart + counts[myId];
  int mycount = myend - mystart;
#endif

#ifdef TARGET_MPI
  gooMalloc((void**) &dev_event_array, dimensions*mycount*sizeof(fptype)); 
  MEMCPY(dev_event_array, host_array + mystart*dimensions, dimensions*mycount*sizeof(fptype), cudaMemcpyHostToDevice);
  delete[] host_array; 

  setNumPerTask(this, mycount);

  delete [] counts;
  delete [] displacements;
#else
  gooMalloc((void**) &dev_event_array, dimensions*numEntries*sizeof(fptype)); 
  MEMCPY(dev_event_array, host_array, dimensions*numEntries*sizeof(fptype), cudaMemcpyHostToDevice);
#endif
}

__host__ void PdfBase::generateNormRange () {
  if (normRanges) gooFree (normRanges);
  gooMalloc((void**)&normRanges, 3*observables.size ()*sizeof (fptype));


  fptype *host_norms = new fptype[3*observables.size()];
  // Don't use index in this case to allow for, eg, 
  // a single observable whose index is 1; or two observables with indices
  // 0 and 2. Make one array per functor, as opposed to variable, to make
  // it easy to pass MetricTaker a range without worrying about which parts
  // to use. 

  int counter = 0;

  //(brad) This is modified since this is added after initial setup.  everything else afterwards needs to propogate these changes
  for (obsIter v = obsBegin(); v != obsEnd(); ++v) {
    host_norms[3*counter + 0] = (*v)->lowerlimit;
    host_norms[3*counter + 1] = (*v)->upperlimit;
    host_norms[3*counter + 2] = integrationBins > 0 ? integrationBins : (*v)->numbins;

    //printf ("l:%f u:%f bins:%f\n", host_params[normalisationIdx + 1], host_params[normalisationIdx + 2], host_params[normalisationIdx + 3]);
    counter ++;
  }

  MEMCPY(normRanges, host_norms, 3*observables.size()*sizeof(fptype), cudaMemcpyHostToDevice);
}

void PdfBase::clearCurrentFit () {
  totalParams = 0; 
  gooFree(dev_event_array);
  dev_event_array = 0; 
}

__host__ void PdfBase::printProfileInfo (bool topLevel) {
#ifdef PROFILING
  if (topLevel) {
    cudaError_t err = MEMCPY_FROM_SYMBOL(host_timeHist, timeHistogram, 10000*sizeof(fptype), 0);
    if (cudaSuccess != err) {
      std::cout << "Error on copying timeHistogram: " << cudaGetErrorString(err) << std::endl;
      return;
    }
    
    std::cout << getName() << " : " << getFunctionIndex() << " " << host_timeHist[100*getFunctionIndex() + getParameterIndex()] << std::endl; 
    for (unsigned int i = 0; i < components.size(); ++i) {
      components[i]->printProfileInfo(false); 
    }
  }
#endif
}



gooError gooMalloc (void** target, size_t bytes) {
// Thrust 1.7 will make the use of THRUST_DEVICE_BACKEND an error
#if THRUST_DEVICE_BACKEND==THRUST_DEVICE_BACKEND_OMP || THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_OMP
  target[0] = malloc(bytes);
  if (target[0]) return gooSuccess;
  else return gooErrorMemoryAllocation; 
#else
  cudaError err = cudaMalloc (target, bytes);
  if (cudaSuccess != err)
  {
    printf ("Cuda malloc error in %s:%i - %s\n", __FILE__, __LINE__, cudaGetErrorString (err));
    return gooErrorMemoryAllocation;
  }

  return gooSuccess; 
#endif
}

gooError gooFree (void* ptr) {
// Thrust 1.7 will make the use of THRUST_DEVICE_BACKEND an error
#if THRUST_DEVICE_BACKEND==THRUST_DEVICE_BACKEND_OMP || THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_OMP
  free(ptr);
  return gooSuccess;
#else
  return (gooError) cudaFree(ptr); 
#endif
}
