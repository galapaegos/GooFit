include Makefile.config

PWD = $(shell /bin/pwd)
SRCDIR = $(PWD)/PDFs

# GooPdf must be first in CUDAglob, as it defines global variables.
FUNCTORLIST    = $(SRCDIR)/GooPdf.cu $(SRCDIR)/AddPdf.cu $(SRCDIR)/GaussianPdf.cu $(SRCDIR)/PolynomialPdf.cu $(SRCDIR)/TddpPdf.cu $(SRCDIR)/DalitzPlotPdf.cu $(SRCDIR)/DalitzVetoPdf.cu $(SRCDIR)/ResonancePdf.cu $(SRCDIR)/MixingTimeResolution_Aux.cu
#FUNCTORLIST    += $(filter-out $(SRCDIR)/GooPdf.cu, $(wildcard $(SRCDIR)/*Pdf.cu))
#FUNCTORLIST   += $(wildcard $(SRCDIR)/*Aux.cu)
HEADERLIST     = $(patsubst %.cu,%.hh,$(SRCFILES))
WRKFUNCTORLIST = $(patsubst $(SRCDIR)/%.cu,wrkdir/%.cu,$(FUNCTORLIST))
#NB, the above are used in the SRCDIR Makefile.

THRUSTO		= wrkdir/Variable.o wrkdir/FitManager.o wrkdir/GooPdfCUDA.o wrkdir/Faddeeva.o wrkdir/FitControl.o wrkdir/PdfBase.o wrkdir/DataSet.o wrkdir/BinnedDataSet.o wrkdir/UnbinnedDataSet.o wrkdir/FunctorWriter.o 
ROOTRIPDIR	= $(PWD)/rootstuff
ROOTRIPOBJS	= $(ROOTRIPDIR)/TMinuit.o $(ROOTRIPDIR)/TRandom.o $(ROOTRIPDIR)/TRandom3.o 
ROOTUTILLIB	= $(ROOTRIPDIR)/libRootUtils.so 

.SUFFIXES: 
.PHONY:		goofit clean 

goofit:		$(THRUSTO) $(ROOTUTILLIB) 
		@echo "Built GooFit objects" 

# One rule for GooFit objects.
wrkdir/%.o:	%.cc %.hh 
		@mkdir -p wrkdir 
		$(CXX) $(INCLUDES) $(CXXFLAGS) $(DEFINEFLAGS) -c -o $@ $<

# A different rule for user-level objects. Notice ROOT_INCLUDES. 
%.o:	%.cu
	$(CXX) $(INCLUDES) $(ROOT_INCLUDES) $(DEFINEFLAGS) $(CXXFLAGS) -c -o $@ $<

# Still a third rule for the ROOT objects - these have their own Makefile. 
$(ROOTRIPDIR)/%.o:	$(ROOTRIPDIR)/%.cc 
			rm -f $@ 
			@echo "Postponing $@ for separate Makefile" 

$(ROOTUTILLIB):	$(ROOTRIPOBJS)
		@cd rootstuff; $(MAKE) 

include $(SRCDIR)/Makefile 

FitManager.o:		FitManager.cc FitManager.hh wrkdir/ThrustFitManagerCUDA.o Variable.o 
			$(CXX) $(DEFINEFLAGS) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

wrkdir/GooPdfCUDA.o:	wrkdir/CUDAglob.cu PdfBase.cu 
			$(CXX) $(CXXFLAGS) $(INCLUDES) -I. $(DEFINEFLAGS) -c $< -o $@ 
			@echo "$@ done"

clean:
		@rm -f *.o wrkdir/*
		cd rootstuff; $(MAKE) clean 
