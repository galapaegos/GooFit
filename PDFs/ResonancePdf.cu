#include "ResonancePdf.hh" 

EXEC_TARGET fptype twoBodyCMmom (double rMassSq, fptype d1m, fptype d2m) {
  // For A -> B + C, calculate momentum of B and C in rest frame of A. 
  // PDG 38.16.

  //fptype kin1 = 1 - POW(d1m+d2m, 2) / rMassSq;
  //if (kin1 >= 0) kin1 = SQRT(kin1);
  //else kin1 = 1;
  //fptype kin2 = 1 - POW(d1m-d2m, 2) / rMassSq;
  //if (kin2 >= 0) kin2 = SQRT(kin2);
  //else kin2 = 1; 

  //return 0.5*SQRT(rMassSq)*kin1*kin2; 

  fptype dpd = d1m + d2m;
  fptype dmd = d1m - d2m;
  fptype irms = 1.0/rMassSq;
  fptype sq = SQRT(rMassSq);
  fptype dpd2 = dpd*dpd;
  fptype dmd2 = dmd*dmd;

  fptype kin1 = 1 - dpd2 * irms;
  fptype kin2 = 1 - dmd2 * irms;
  fptype sqkin1 = SQRT(kin1);
  fptype sqkin2 = SQRT(kin2);

  kin1 = (kin1 >= 0.0) ? sqkin1 : 1.0;
  kin2 = (kin2 >= 0.0) ? sqkin2 : 1.0;

  return 0.5*sq*kin1*kin2; 
}


EXEC_TARGET fptype dampingFactorSquare (fptype cmmom, int spin, fptype mRadius) {
  fptype square = mRadius*mRadius*cmmom*cmmom;
  fptype dfsq = 1 + square; // This accounts for spin 1
  if (2 == spin) dfsq += 8 + 2*square + square*square; // Coefficients are 9, 3, 1.   

  // Spin 3 and up not accounted for. 
  return dfsq; 
}

EXEC_TARGET fptype spinFactor (unsigned int spin, fptype motherMass, fptype daug1Mass, fptype daug2Mass, fptype daug3Mass, fptype m12, fptype m13, fptype m23, unsigned int cyclic_index) {
  if (0 == spin) return 1; // Should not cause branching since every thread evaluates the same resonance at the same time. 
  /*
  // Copied from BdkDMixDalitzAmp
   
  fptype _mA = (PAIR_12 == cyclic_index ? daug1Mass : (PAIR_13 == cyclic_index ? daug1Mass : daug3Mass)); 
  fptype _mB = (PAIR_12 == cyclic_index ? daug2Mass : (PAIR_13 == cyclic_index ? daug3Mass : daug3Mass)); 
  fptype _mC = (PAIR_12 == cyclic_index ? daug3Mass : (PAIR_13 == cyclic_index ? daug2Mass : daug1Mass)); 
    
  fptype _mAC = (PAIR_12 == cyclic_index ? m13 : (PAIR_13 == cyclic_index ? m12 : m12)); 
  fptype _mBC = (PAIR_12 == cyclic_index ? m23 : (PAIR_13 == cyclic_index ? m23 : m13)); 
  fptype _mAB = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23)); 

  // The above, collapsed into single tests where possible. 
  fptype _mA = (PAIR_13 == cyclic_index ? daug3Mass : daug2Mass);
  fptype _mB = (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass); 
  fptype _mC = (PAIR_12 == cyclic_index ? daug3Mass : (PAIR_13 == cyclic_index ? daug2Mass : daug1Mass)); 

  fptype _mAC = (PAIR_23 == cyclic_index ? m13 : m23);
  fptype _mBC = (PAIR_12 == cyclic_index ? m13 : m12);
  fptype _mAB = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23)); 
  */

  // Copied from EvtDalitzReso, with assumption that pairAng convention matches pipipi0 from EvtD0mixDalitz.
  // Again, all threads should get the same branch. 
  fptype _mA = (PAIR_12 == cyclic_index ? daug1Mass : (PAIR_13 == cyclic_index ? daug3Mass : daug2Mass));
  fptype _mB = (PAIR_12 == cyclic_index ? daug2Mass : (PAIR_13 == cyclic_index ? daug1Mass : daug3Mass));
  fptype _mC = (PAIR_12 == cyclic_index ? daug3Mass : (PAIR_13 == cyclic_index ? daug2Mass : daug1Mass));
  fptype _mAC = (PAIR_12 == cyclic_index ? m13 : (PAIR_13 == cyclic_index ? m23 : m12)); 
  fptype _mBC = (PAIR_12 == cyclic_index ? m23 : (PAIR_13 == cyclic_index ? m12 : m13)); 
  fptype _mAB = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23)); 

  fptype massFactor = 1.0/_mAB;
  fptype sFactor = -1; 
  sFactor *= ((_mBC - _mAC) + (massFactor*(motherMass*motherMass - _mC*_mC)*(_mA*_mA-_mB*_mB)));
  if (2 == spin) {
    sFactor *= sFactor; 
    fptype extraterm = ((_mAB-(2*motherMass*motherMass)-(2*_mC*_mC))+massFactor*pow((motherMass*motherMass-_mC*_mC),2));
    extraterm *= ((_mAB-(2*_mA*_mA)-(2*_mB*_mB))+massFactor*pow((_mA*_mA-_mB*_mB),2));
    extraterm /= 3;
    sFactor -= extraterm;
  }
  return sFactor; 
}

EXEC_TARGET devcomplex<fptype> plainBW (fptype m12, fptype m13, fptype m23, unsigned int* indices)
{
  int numParams = cudaArray[*indices];
  //these are + 1, which is where the elements start. 
  fptype resmass                = cudaArray[*indices + 3];
  fptype reswidth               = cudaArray[*indices + 4];
  
  int numObs = cudaArray[*indices + numParams + 1];
  
  int numCons = cudaArray[*indices + numParams + 1 + numObs + 1];
  int consIdx = numParams + 1 + numObs + 2;
  
  unsigned int spin             = cudaArray[*indices + consIdx + 0];
  unsigned int cyclic_index     = cudaArray[*indices + consIdx + 1]; 
  
  fptype motherMass             = cudaArray[*indices + consIdx + 2];
  fptype daug1Mass              = cudaArray[*indices + consIdx + 3];
  fptype daug2Mass              = cudaArray[*indices + consIdx + 4];
  fptype daug3Mass              = cudaArray[*indices + consIdx + 5];
  fptype meson_radius           = cudaArray[*indices + consIdx + 6];

  fptype rMassSq = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
  fptype frFactor = 1;

  resmass *= resmass; 
  // Calculate momentum of the two daughters in the resonance rest frame; note symmetry under interchange (dm1 <-> dm2). 
  fptype measureDaughterMoms = twoBodyCMmom(rMassSq, 
					    (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), 
					    (PAIR_12 == cyclic_index ? daug2Mass : daug3Mass));
  fptype nominalDaughterMoms = twoBodyCMmom(resmass, 
					    (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), 
					    (PAIR_12 == cyclic_index ? daug2Mass : daug3Mass));

  if (0 != spin) {
    frFactor =  dampingFactorSquare(nominalDaughterMoms, spin, meson_radius);
    frFactor /= dampingFactorSquare(measureDaughterMoms, spin, meson_radius); 
  }  
 
  // RBW evaluation
  fptype A = (resmass - rMassSq); 
  fptype B = resmass*reswidth * POW(measureDaughterMoms / nominalDaughterMoms, 2.0*spin + 1) * frFactor / SQRT(rMassSq);
  fptype C = 1.0 / (A*A + B*B); 
  devcomplex<fptype> ret(A*C, B*C); // Dropping F_D=1

  ret *= SQRT(frFactor); 
  fptype spinF = spinFactor(spin, motherMass, daug1Mass, daug2Mass, daug3Mass, m12, m13, m23, cyclic_index); 
  ret *= spinF; 

  return ret; 
}

EXEC_TARGET devcomplex<fptype> gaussian (fptype m12, fptype m13, fptype m23, unsigned int* indices) {
  // indices[1] is unused constant index, for consistency with other function types. 
  fptype resmass                = cudaArray[indices[2]];
  fptype reswidth               = cudaArray[indices[3]];
  unsigned int cyclic_index     = indices[4]; 

  // Notice sqrt - this function uses mass, not mass-squared like the other resonance types. 
  fptype massToUse = SQRT(PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
  massToUse -= resmass;
  massToUse /= reswidth;
  massToUse *= massToUse;
  fptype ret = EXP(-0.5*massToUse); 

  // Ignore factor 1/sqrt(2pi). 
  ret /= reswidth;

  return devcomplex<fptype>(ret, 0); 
}

EXEC_TARGET fptype hFun (double s, double daug2Mass, double daug3Mass) {
  // Last helper function
  const fptype _pi = 3.14159265359;
  double sm   = daug2Mass + daug3Mass;
  double SQRTs = sqrt(s);
  double k_s = twoBodyCMmom(s, daug2Mass, daug3Mass);

  double val = ((2/_pi) * (k_s/SQRTs) * log( (SQRTs + 2*k_s)/(sm)));

  return val;
}

EXEC_TARGET fptype dh_dsFun (double s, double daug2Mass, double daug3Mass) {
  // Yet another helper function
  const fptype _pi = 3.14159265359;
  double k_s = twoBodyCMmom(s, daug2Mass, daug3Mass);
  
  double val = (hFun(s, daug2Mass, daug3Mass) * (1.0/(8.0*pow(k_s, 2)) - 1.0/(2.0 * s)) + 1.0/(2.0* _pi*s));
  return val;
}


EXEC_TARGET fptype dFun (double s, double daug2Mass, double daug3Mass) {
  // Helper function used in Gronau-Sakurai
  const fptype _pi = 3.14159265359;
  double sm   = daug2Mass + daug3Mass;
  double sm24 = sm*sm/4.0;
  double m    = sqrt(s);
  double k_m2 = twoBodyCMmom(s, daug2Mass, daug3Mass);
 
  double val = 3.0/_pi * sm24/pow(k_m2, 2) * log((m + 2*k_m2)/sm) + m/(2*_pi*k_m2) - sm24*m/(_pi * pow(k_m2, 3));
  return val;
}

EXEC_TARGET fptype fsFun (double s, double m2, double gam, double daug2Mass, double daug3Mass) {
  // Another G-S helper function
   
  double k_s   = twoBodyCMmom(s,  daug2Mass, daug3Mass);
  double k_Am2 = twoBodyCMmom(m2, daug2Mass, daug3Mass);
   
  double f     = gam * m2 / POW(k_Am2, 3);
  f           *= (POW(k_s, 2) * (hFun(s, daug2Mass, daug3Mass) - hFun(m2, daug2Mass, daug3Mass)) + (m2 - s) * pow(k_Am2, 2) * dh_dsFun(m2, daug2Mass, daug3Mass));
 
  return f;
}

EXEC_TARGET devcomplex<fptype> gouSak (fptype m12, fptype m13, fptype m23, unsigned int* indices) {
  int idx[6];
  idx[1] = indices[1];
  idx[2] = indices[2];
  idx[3] = indices[3];
  idx[4] = indices[4];
  idx[5] = indices[5];

  fptype motherMass             = cudaArray[idx[1]+0];
  fptype daug1Mass              = cudaArray[idx[1]+1];
  fptype daug2Mass              = cudaArray[idx[1]+2];
  fptype daug3Mass              = cudaArray[idx[1]+3];
  fptype meson_radius           = cudaArray[idx[1]+4];

  fptype resmass                = cudaArray[idx[2]];
  fptype reswidth               = cudaArray[idx[3]];
  unsigned int spin             = idx[4];
  unsigned int cyclic_index     = idx[5]; 

  fptype rMassSq = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
  fptype frFactor = 1;

  resmass *= resmass; 
  // Calculate momentum of the two daughters in the resonance rest frame; note symmetry under interchange (dm1 <-> dm2). 
  fptype measureDaughterMoms = twoBodyCMmom(rMassSq, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_12 == cyclic_index ? daug2Mass : daug3Mass));
  fptype nominalDaughterMoms = twoBodyCMmom(resmass, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_12 == cyclic_index ? daug2Mass : daug3Mass));

  if (0 != spin) {
    frFactor =  dampingFactorSquare(nominalDaughterMoms, spin, meson_radius);
    frFactor /= dampingFactorSquare(measureDaughterMoms, spin, meson_radius); 
  }
  
  // Implement Gou-Sak:
  fptype sqrtfrFactor = SQRT(frFactor);

  fptype D = (1.0 + dFun(resmass, daug2Mass, daug3Mass) * reswidth/SQRT(resmass));
  fptype E = resmass - rMassSq + fsFun(rMassSq, resmass, reswidth, daug2Mass, daug3Mass);
  fptype F = SQRT(resmass) * reswidth * POW(measureDaughterMoms / nominalDaughterMoms, 2.0*spin + 1) * frFactor;

  D       /= (E*E + F*F);
  devcomplex<fptype> retur(D*E, D*F); // Dropping F_D=1
  retur *= sqrtfrFactor;
  retur *= spinFactor(spin, motherMass, daug1Mass, daug2Mass, daug3Mass, m12, m13, m23, cyclic_index);

  return retur; 
}


EXEC_TARGET devcomplex<fptype> lass (fptype m12, fptype m13, fptype m23, unsigned int* indices) {
  int idx[6];
  idx[1] = indices[1];
  idx[2] = indices[2];
  idx[3] = indices[3];
  idx[4] = indices[4];
  idx[5] = indices[5];

  fptype motherMass             = cudaArray[idx[1]+0];
  fptype daug1Mass              = cudaArray[idx[1]+1];
  fptype daug2Mass              = cudaArray[idx[1]+2];
  fptype daug3Mass              = cudaArray[idx[1]+3];
  fptype meson_radius           = cudaArray[idx[1]+4];

  fptype resmass                = cudaArray[idx[2]];
  fptype reswidth               = cudaArray[idx[3]];
  unsigned int spin             = idx[4];
  unsigned int cyclic_index     = idx[5];

  fptype rMassSq = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
  fptype frFactor = 1;

  resmass *= resmass;
  // Calculate momentum of the two daughters in the resonance rest frame; note symmetry under interchange (dm1 <-> dm2).
  
  fptype measureDaughterMoms = twoBodyCMmom(rMassSq, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_23 == cyclic_index ? daug3Mass : daug2Mass));
  fptype nominalDaughterMoms = twoBodyCMmom(resmass, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_23 == cyclic_index ? daug3Mass : daug2Mass));

  if (0 != spin) {
    frFactor =  dampingFactorSquare(nominalDaughterMoms, spin, meson_radius);
    frFactor /= dampingFactorSquare(measureDaughterMoms, spin, meson_radius);
  }

  //Implement LASS:
  /*
  fptype s = kinematics(m12, m13, _trackinfo[i]);
  fptype q = twoBodyCMmom(s, _trackinfo[i]);
  fptype m0  = _massRes[i]->getValFast();
  fptype _g0 = _gammaRes[i]->getValFast();
  int spin   = _spinRes[i];
  fptype g = runningWidthFast(s, m0, _g0, spin, _trackinfo[i], FrEval(s, m0, _trackinfo[i], spin));
  */

  fptype q = measureDaughterMoms;
  //fptype g = reswidth * POW(measureDaughterMoms / nominalDaughterMoms, 2.0*spin + 1) * frFactor / SQRT(rMassSq);
  fptype g = reswidth * POW(measureDaughterMoms / nominalDaughterMoms, 2.0*spin + 1) * frFactor * RSQRT(rMassSq);

  fptype _a    = 0.22357;
  fptype _r    = -15.042;
  fptype _R    = 1; // ?
  fptype _phiR = 1.10644;

  fptype _aq   = _a*q;
  fptype _rq   = _r*q;
  fptype _B    = 0.614463;
  fptype _phiB = -0.0981907;

  fptype i_aq  = 1.0/_aq;
  fptype _rqq  = _rq*q;
  fptype h_rq  = 0.5*_rq;
  fptype i_a   = 1.0/_a;

  fptype m_phiB = 2.0*_phiB;
  fptype sqrtResMass = SQRT(resmass);

  // background phase motion
  fptype cot_deltaB = i_aq + h_rq;
  fptype qcot_deltaB = i_a + 0.5*_rqq;

  fptype cos_phiB = cos(_phiB);
  fptype sin_phiB = sin(_phiB);

  fptype sqrtResMassg = sqrtResMass*g;

  fptype _phiRpm_phiB = _phiR + m_phiB;

  devcomplex<fptype> a1 = devcomplex<fptype>(qcot_deltaB, q);
  devcomplex<fptype> a2 = devcomplex<fptype>(qcot_deltaB, -q);

  fptype cos1 = cos(_phiRpm_phiB);
  fptype sin1 = sin(_phiRpm_phiB);

  // calculate resonant part
  //devcomplex<fptype> expi2deltaB = devcomplex<fptype>(qcot_deltaB,q)/devcomplex<fptype>(qcot_deltaB,-q);
  devcomplex<fptype> expi2deltaB = a1/a2;
  //devcomplex<fptype> resT = devcomplex<fptype>(cos(_phiR+2*_phiB),sin(_phiR+2*_phiB))*_R;
  devcomplex<fptype> resT = devcomplex<fptype>(cos1, sin1)*_R;

  fptype resmass_m_rMassSq = resmass - rMassSq;
  fptype resmasswidth = resmass*reswidth;

  //devcomplex<fptype> prop = devcomplex<fptype>(1, 0)/devcomplex<fptype>(resmass-rMassSq, SQRT(resmass)*g);
  devcomplex<fptype> prop = devcomplex<fptype>(1, 0)/devcomplex<fptype>(resmass_m_rMassSq, sqrtResMassg);
  // resT *= prop*m0*_g0*m0/twoBodyCMmom(m0*m0, _trackinfo[i])*expi2deltaB;

  devcomplex<fptype> b1 (cos_phiB, sin_phiB);
  devcomplex<fptype> propdeltaB = prop*expi2deltaB;
  fptype cot_deltaB_sin = cot_deltaB*sin_phiB;

  resT *= propdeltaB*(resmasswidth/nominalDaughterMoms);

  // calculate bkg part
  //resT += devcomplex<fptype>(cos(_phiB),sin(_phiB))*_B*(cos(_phiB)+cot_deltaB*sin(_phiB))*SQRT(rMassSq)/devcomplex<fptype>(qcot_deltaB,-q);
  resT += b1*_B*(cos_phiB + cot_deltaB_sin)*SQRT(rMassSq)/a2;

  resT *= SQRT(frFactor);
  resT *= spinFactor(spin, motherMass, daug1Mass, daug2Mass, daug3Mass, m12, m13, m23, cyclic_index);
  
  return resT;
}


EXEC_TARGET devcomplex<fptype> nonres (fptype m12, fptype m13, fptype m23, unsigned int* indices) {
  return devcomplex<fptype>(1, 0); 
}


EXEC_TARGET void getAmplitudeCoefficients (devcomplex<fptype> a1, devcomplex<fptype> a2, fptype& a1sq, fptype& a2sq, fptype& a1a2real, fptype& a1a2imag) {
  // Returns A_1^2, A_2^2, real and imaginary parts of A_1A_2^*
  a1sq = a1.abs2();
  a2sq = a2.abs2();
  a1 *= conj(a2);
  a1a2real = a1.real;
  a1a2imag = a1.imag; 
}

MEM_DEVICE resonance_function_ptr ptr_to_RBW = plainBW;
MEM_DEVICE resonance_function_ptr ptr_to_GOUSAK = gouSak; 
MEM_DEVICE resonance_function_ptr ptr_to_GAUSSIAN = gaussian;
MEM_DEVICE resonance_function_ptr ptr_to_NONRES = nonres;
MEM_DEVICE resonance_function_ptr ptr_to_LASS = lass;

ResonancePdf::ResonancePdf (string name, 
						Variable* ar, 
						Variable* ai, 
						Variable* mass, 
						Variable* width, 
						unsigned int sp, 
						unsigned int cyc) 
  : GooPdf(0, name)
  , amp_real(ar)
  , amp_imag(ai), bw (true), gs (false), lass (false), nr (false), gauss (false)
{
  vector<unsigned int> pindices; 
  pindices.push_back(0); 
  // Making room for index of decay-related constants. Assumption:
  // These are mother mass and three daughter masses in that order.
  // They will be registered by the object that uses this resonance,
  // which will tell this object where to find them by calling setConstantIndex. 

  pindices.push_back(registerParameter(mass));
  pindices.push_back(registerParameter(width)); 
  pindices.push_back(sp);
  pindices.push_back(cyc); 

  untracked.push_back (ar);
  untracked.push_back (ai);
  //parameterList.push_back (mass);
  //parameterList.push_back (width);

  constants.push_back (sp);
  constants.push_back (cyc);

  GET_FUNCTION_ADDR(ptr_to_RBW);
  initialise(pindices); 
}

ResonancePdf::ResonancePdf (string name, 
						Variable* ar, 
						Variable* ai, 
						unsigned int sp, 
						Variable* mass, 
						Variable* width, 
						unsigned int cyc) 
  : GooPdf(0, name)
  , amp_real(ar)
  , amp_imag(ai), bw (false), gs (true), lass (false), nr (false), gauss (false)
{
  // Same as BW except for function pointed to. 
  vector<unsigned int> pindices; 
  pindices.push_back(0); 
  pindices.push_back(registerParameter(mass));
  pindices.push_back(registerParameter(width)); 
  pindices.push_back(sp);
  pindices.push_back(cyc); 

  untracked.push_back (ar);
  untracked.push_back (ai);
  //parameterList.push_back (mass);
  //parameterList.push_back (width);

  constants.push_back (sp);
  constants.push_back (cyc);

  GET_FUNCTION_ADDR(ptr_to_GOUSAK);
  initialise(pindices); 
} 
 
   
ResonancePdf::ResonancePdf (string name,
                                                Variable* ar,
                                                Variable* ai,
						Variable* mass,
                                                unsigned int sp,
                                                Variable* width,
                                                unsigned int cyc)
  : GooPdf(0, name)
  , amp_real(ar)
  , amp_imag(ai), bw (false), gs (false), lass (true), nr (false), gauss (false)
{
  // Same as BW except for function pointed to.
  vector<unsigned int> pindices;
  pindices.push_back(0);
  pindices.push_back(registerParameter(mass));
  pindices.push_back(registerParameter(width));
  pindices.push_back(sp);
  pindices.push_back(cyc);

  untracked.push_back (ar);
  untracked.push_back (ai);
  //parameterList.push_back (mass);
  //parameterList.push_back (width);

  constants.push_back (sp);
  constants.push_back (cyc);

  GET_FUNCTION_ADDR(ptr_to_LASS);
  initialise(pindices);
}


ResonancePdf::ResonancePdf (string name, 
						Variable* ar, 
						Variable* ai) 
  : GooPdf(0, name)
  , amp_real(ar)
  , amp_imag(ai), bw (false), gs (false), lass (false), nr (true), gauss (false)
{
  vector<unsigned int> pindices; 
  pindices.push_back(0); 
  // Dummy index for constants - won't use it, but calling 
  // functions can't know that and will call setConstantIndex anyway. 
  untracked.push_back (ar);
  untracked.push_back (ai);
  untracked.push_back (ar);
  untracked.push_back (ai);
  //untracked.push_back (ar);
  //untracked.push_back (ai);
  
  constants.push_back (0);
  constants.push_back (0);
  
  GET_FUNCTION_ADDR(ptr_to_NONRES);
  initialise(pindices); 
}

ResonancePdf::ResonancePdf (string name,
						Variable* ar, 
						Variable* ai,
						Variable* mean, 
						Variable* sigma,
						unsigned int cyc) 
  : GooPdf(0, name)
  , amp_real(ar)
  , amp_imag(ai), bw (false), gs (false), lass (false), nr (false), gauss (true)
{
  vector<unsigned int> pindices; 
  pindices.push_back(0); 
  // Dummy index for constants - won't use it, but calling 
  // functions can't know that and will call setConstantIndex anyway. 
  pindices.push_back(registerParameter(mean));
  pindices.push_back(registerParameter(sigma)); 
  pindices.push_back(cyc); 

  untracked.push_back (ar);
  untracked.push_back (ai);
  //parameterList.push_back (mean);
  //parameterList.push_back (sigma);

  constants.push_back (cyc);
  constants.push_back (0);

  GET_FUNCTION_ADDR(ptr_to_GAUSSIAN);
  initialise(pindices); 

}

__host__ void ResonancePdf::copyParams (std::vector<Variable*> vars)
{
  //(brad) copy all values into host_params to be transfered
  //note these are indexed the way they are passed.
  // Copies values of Variable objects

  //copy from vars to our local copy, should be able to remove this at some point(?)
  //for (int x = 0; x < vars.size (); x++)
  //{
  //  for (int y = 0; y < parameterList.size (); y++)
  //  {
  //    if (parameterList[y]->name == vars[x]->name)
  //    {
  //      parameterList[y]->value = vars[x]->value;
  //      parameterList[y]->blind = vars[x]->blind;
  //    }
  //  }
  //}

  int counter = untracked.size ();
  for (int i = 0; i < parameterList.size (); i++)
    host_params[parametersIdx + counter++] = parameterList[i]->value + parameterList[i]->blind;

  //recurse
  for (int i = 0; i < components.size (); i++)
    components[i]->copyParams(vars);
}

__host__ void ResonancePdf::recursiveSetIndices ()
{
  //(brad): copy into our device list, will need to have a variable to determine type
  if (bw)
    GET_FUNCTION_ADDR(ptr_to_RBW);
  else if (gs)
    GET_FUNCTION_ADDR(ptr_to_GOUSAK);
  else if (lass)
    GET_FUNCTION_ADDR(ptr_to_LASS);
  else if (nr)
    GET_FUNCTION_ADDR(ptr_to_NONRES);
  else if (gauss)
    GET_FUNCTION_ADDR(ptr_to_GAUSSIAN);

  host_function_table[num_device_functions] = host_fcn_ptr;
  functionIdx = num_device_functions;
  num_device_functions ++;
 //(brad): confused by this parameters variable.  Wouldn't each PDF get the current total, not the current amount?
  //parameters = totalParams;
  //totalParams += (2 + pindices.size() + observables.size());

  //(brad)additional vector to keep track of variables that are not constant
  //in order to figure out the next index, we will need to do some additions to get all the proper offsets
  host_params[totalParams++] = untracked.size () + parameterList.size ();
  parametersIdx = totalParams;
  for (int i = 0; i < untracked.size (); i++)
    host_params[totalParams++] = untracked[i]->value;
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
}

__host__ void ResonancePdf::setDecayInfo (fptype mom, fptype d1, fptype d2, fptype d3, fptype mr)
{
  constants.push_back (mom);
  constants.push_back (d1);
  constants.push_back (d2);
  constants.push_back (d3);
  constants.push_back (mr);
}
