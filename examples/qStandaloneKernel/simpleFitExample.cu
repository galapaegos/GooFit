#include "FitManager.hh"
#include "UnbinnedDataSet.hh" 
#include "GPdf.hh"

#include "TRandom.hh" 
#include "fakeTH1F.h"

#include <sys/time.h>
#include <sys/times.h>
#include <iostream>

#include <chrono>

#define timeCheck() std::chrono::high_resolution_clock::now ()
#define duration(x) std::chrono::duration<double, milli> (x).count()
#define profile(x)\
 {\
   auto start = std::chrono::high_resolution_clock::now();\
   x;\
   auto stop = std::chrono::high_resolution_clock::now();\
   std::cout << std::chrono::duration<double, milli> (stop - start).count() << " ms. " << std::endl;\
 }

void fitAndPlot (GPdf* total, UnbinnedDataSet* data, TH1F& dataHist, Variable* xvar, const char* fname)
{
  total->setData(data);
  FitManager fitter(total);
  profile (fitter.fit());
  fitter.getMinuitValues(); 

/*
  TH1F pdfHist("pdfHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfHist.SetStats(false);

  UnbinnedDataSet grid(xvar);
  double step = (xvar->upperlimit - xvar->lowerlimit)/xvar->numbins;
  for (int i = 0; i < xvar->numbins; ++i) {
    xvar->value = xvar->lowerlimit + (i + 0.5) * step;
    grid.addEvent();
  }

  total->setData(&grid);
  vector<vector<double> > pdfVals;
  total->getCompProbsAtDataPoints(pdfVals);

  double totalPdf = 0;
  for (int i = 0; i < grid.getNumEvents(); ++i) {
    grid.loadEvent(i);
    pdfHist.Fill(xvar->value, pdfVals[0][i]);
    totalPdf += pdfVals[0][i];
  }

  for (int i = 0; i < xvar->numbins; ++i) {
    double val = pdfHist.GetBinContent(i+1);
    val /= totalPdf;
    val *= data->getNumEvents();
    pdfHist.SetBinContent(i+1, val);
  }
  
  if (dataHist.GetNumbins() == pdfHist.GetNumbins()) {
     for (int i = 0; i < dataHist.GetNumbins(); ++i) {
        std::cout << dataHist.GetBinCenter(i+1) << " "
                  << dataHist.GetBinContent(i+1) << " "
                  << pdfHist.GetBinContent(i+1)
                  << std::endl;
     }
  } else {
     std::cerr << "I don't understand dataHist/pdfHist" << std::endl;
  }
*/
}

int main (int argc, char** argv)
{
  auto start = timeCheck();
  if (argc != 2)
  {
    printf ("Need to pass number of events to generate\n");
    return -1;
  }

  int numEvents = atoi (argv[1]);

  // Independent variable. 
  Variable* xvar = new Variable("xvar", -100, 100); 
  xvar->numbins = 1000; // For such a large range, want more bins for better accuracy in normalisation. 

  // Data sets for the three fits. 
  UnbinnedDataSet landdata(xvar);

  // Histograms for showing the fit. 
  TH1F landHist("landHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  landHist.SetStats(false); 

  TRandom donram(42); 

  double leftSigma = 13;
  double rightSigma = 29;
  double leftIntegral = 0.5 / (leftSigma * sqrt(2*M_PI));
  double rightIntegral = 0.5 / (rightSigma * sqrt(2*M_PI));
  double totalIntegral = leftIntegral + rightIntegral; 
  //double bifpoint = -10; 

  // Generating toy MC. 
  for (int i = 0; i < numEvents; ++i) {
    // Landau
    xvar->value = xvar->upperlimit + 1; 
    while ((xvar->value > xvar->upperlimit) || (xvar->value < xvar->lowerlimit)) {
      xvar->value = donram.Landau(20, 1); 
    }
    landdata.addEvent(); 
    landHist.Fill(xvar->value); 
  }

  Variable* mpv            = new Variable("mpv", 40, 0, 150);
  Variable* sigma          = new Variable("sigma", 5, 0, 30);
  //GooPdf* landau = new LandauPdf("landau", xvar, mpv, sigma); 
  GPdf *landau = new GPdf(xvar, "landau", mpv, sigma);
  fitAndPlot(landau, &landdata, landHist, xvar, "landau.png"); 

  auto end = timeCheck(); 

  std::cout << duration(end - start) << "ms" << std::endl;

  return 0;
}
