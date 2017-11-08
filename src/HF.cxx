#include "HF.h"
#include "Grid.h"
#include "Orbital.h"

#include <stdexcept>
#include <exception>

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <cmath>
#include <iostream>
#include <iomanip>

#include "utils.h"

HF::HF(const Grid &g, double Z)
  : _g(g), _Z(Z) {
  _pot.resize(_g.N());
  _potIndep.resize(_g.N());
  for (int k = 0; k < _g.N(); ++k) {
    _pot[k] = -_Z/_g(k);
    _potIndep[k] = 0;
  }
  _gamma_scf = 0.7;
}

HF::~HF() {
}

std::vector<double> HF::getOrbital(int no, int lo, int mo) {
  Orbital &o = _o[no];
  std::vector<double> res;
  for (int k = 0; k < _g.N(); ++k) {
    res.push_back(o.getNorm(k, lo, mo, _g));
  }
  return res;
}

void HF::gammaSCF(double g) {
  _gamma_scf = g;
}

void HF::solve(int NiterSCF, int Niter, double F0stop) {
  _dE.resize(_o.size());
  int nStepSCF = 0;
  while (nStepSCF < NiterSCF) {
    std::cout << "SCF step " << nStepSCF << std::endl;
    solveForFixedPotentials(Niter, F0stop);
    nStepSCF++;
    calculateVd(_gamma_scf);
  }
}

// direct potential calculation
// V = int_Oa int_ra rpsi1(ra) rpsi2(ra) Yl1m1(Oa) Yl2m2(Oa)/|ra-rb| ra^2 dOa dra
// with 1 = 2 -> k1
// 1/|ra - rb| = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1) r<^l/r>^(l+1) Y*lm(Oa) Ylm(Ob)
// V = \sum_l=0^inf \sum_m=-l^l ( int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra ) (int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa) Ylm(Ob)
// beta(rb, l) = int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra
// T1 = int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa
// T2 = Ylm(Ob)
//
// We multiply by Y*lomo(Ob) and take the existing Yljmj(Ob) from the orbital and integrate in dOb to get the radial equations
// int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^m int Ylm Ylo(-mo) Yljmj dOb = sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
//
//
// V = \sum_m \sum_l=0^inf beta(rb, l) T1(l, m) T2(l, m)
//
//
// T1 = int Yl1m1 Yl1m1 Y*lm = (-1)**m int Yl1m1 Yl1m1 Yl(-m)
// T1 = (-1)**m*(-1)**m*np.sqrt((2*l1+1)*(2*l1+1)/(4*np.pi*(2*l+1)))*CG(l1,l1,0,0,l,0)*CG(l1,l1,m1,m1,l,-(-m))
//
// T2 = 1.0/(4*np.pi) int Ylm dOb
//
void HF::calculateVd(double gamma) {
  // we are looking to calculate Vd(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_k1(r1) Y_k1(O)*Y_k1(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
    for (int lj = 0; lj < _o[ko].L()+1; ++lj) { // loop over l in Y_j
      for (int mj = -lj; mj < lj+1; ++mj) { // loop over m in Y_j

        for (auto &vdLm : _vd[ko]) { // calculate a term in square brackets above
          int vdl = vdLm.first.first; // these are the l and m for Y*_i
          int vdm = vdLm.first.second;
          if (vdl != lj || vdm != mj) continue;
          std::vector<double> &currentVd = vdLm.second; // this is the r-dependent part

          std::vector<double> vd(_g.N(), 0); // calculate it here first
          // loop over orbitals (this is the sum over k1 above)
          for (int k1 = 0; k1 < _o.size(); ++k1) {

            for (int l1 = 0; l1 < _o[k1].L()+1; ++l1) { // each orbital in the sum is actually orb_ko = sum_l1,m1 rpsi_l1,m1 Y_l1m1, so loop over this sum
              for (int m1 = -l1; m1 < l1+1; ++m1) {

                // now actually calculate it from the expansion above
                int lmax = 2;
                for (int l = 0; l < lmax+1; ++l) {
                  for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
                    double beta = 0;
                    double r2 = _g(ir2);
                    for (int ir1 = 0; ir1 < _g.N(); ++ir1) {
                      double r1 = _g(ir1);
                      double dr = 0;
                      if (ir1 < _g.N()-1) dr = _g(ir1+1) - _g(ir1);
                      double rs = r1;
                      double rb = r2;
                      if (rb < rs) {
                        rs = r2;
                        rb = r1;
                      }
                      beta += 4*M_PI/(2.0*l + 1.0)*std::pow(_o[k1].getNorm(ir1, l1, m1, _g), 2)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
                    }
                    double T = 0;
                    for (int m = -l; m < l + 1; ++m) {
                      double T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                      // int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^mo int Ylm Ylo(-mo) Yljmj dOb = (-1)^(mo+mj) sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
                      double T2 = 0;
                      T2 = std::pow(-1, vdm+mj)*std::sqrt((2*l+1)*(2*vdl+1)/(4*M_PI*(2*lj+1)))*CG(l, vdl, 0, 0, lj, 0)*CG(l, vdl, m, -vdm, lj, -mj);
                      T += T1*T2;
                    }
                    vd[ir2] += beta*T;
                  } // for each r in Vd integration
                } // for each l in the 1/|r1 - r2| expansion in sph. harm.
              } // for each m1 of the orbital basis expansion
            } // for each l1 of the orbital basis expansion

          } // for each orbital in the Vd sum

          // for a test in He
          for (int k = 0; k < _g.N(); ++k) vd[k] *= 0.5;

          for (int k = 0; k < _g.N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*vd[k];

        } // for each Vd term in a unique equation coming from the multiplication by Y*_i
      } // for each m in the orbital basis expansion
    } // for each l in the orbital basis expansion

  }
}

std::vector<double> HF::getNucleusPotential() {
  return _pot;
}

std::vector<double> HF::getDirectPotential() {
  return _vd[0][std::pair<int, int>(0, 0)];
}

void HF::solveForFixedPotentials(int Niter, double F0stop) {
  double gamma = 0.5; // move in the direction of the negative slope with this velocity per step

  long double F = 0;
  int nStep = 0;
  while (nStep < Niter) {
    // compute sum of squares of F(x_old)
    nStep += 1;
    F = step();

    // limit maximum energy step in a single direction to be 0.1*gamma
    double gscale = 1;

    // change orbital energies
    std::cout << "Orbital energies at step " << nStep << ", with constraint = " << std::setw(16) << F << "." << std::endl;
    std::cout << std::setw(5) << "Index" << " " << std::setw(16) << "Energy (H)" << " " << std::setw(16) << "next energy (H) " << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      double stepdE = gscale*gamma*_dE[k];
      double newE = (_o[k].E()+stepdE);
      std::cout << std::setw(5) << k << " " << std::setw(16) << std::setprecision(14) << _o[k].E() << " " << std::setw(16) << std::setprecision(16) << newE << std::endl;
      _o[k].E(newE);
    }

    if (std::fabs(F) < F0stop) break;
  }
}

// solve for a fixed energy and calculate _dE for the next step
long double HF::step() {
  // TODO: Ignore off-diagonal entries due to vxc now ... will add iteratively later
  // https://ocw.mit.edu/courses/mathematics/18-409-topics-in-theoretical-computer-science-an-algorithmists-toolkit-fall-2009/lecture-notes/MIT18_409F09_scribe21.pdf

  long double F = 0;
  for (int k = 0; k < _o.size(); ++k) {
    double E = _o[k].E();

    int lmain = _o[k].initialL();
    int mmain = _o[k].initialM();
    // calculate crossing of potential at zero for lmain,mmain
    int icl = 0;
    double a_m1 = 0;
    for (int i = 1; i < _g.N(); ++i) {
      double r = _g(i);
      double a = 2*std::pow(r, 2)*(E - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i]) - std::pow(lmain + 0.5, 2);
      if (a*a_m1 < 0) {
        icl = i;
        break;
      }
      a_m1 = a;
    }

    long double dE = 0.1;
    long double F1 = 0;
    long double F2 = 0;
    int n1 = 0;
    int n2 = 0;
    F1 = solveOrbitalFixedEnergy(E+dE, k, icl, n1);
    F2 = solveOrbitalFixedEnergy(E, k, icl, n2);
    if (F2 != F1) {
      _dE[k] = -F2*dE/(F1 - F2);
    } else {
      _dE[k] = dE;
    }
    F += F2;
  }
  return F;
}

void HF::solveInward(double E, int n, int l, std::vector<double> &solution, std::vector<double> &f) {
  int N = _g.N();
  solution[N-1] = 0;//std::exp(-std::sqrt(-2*E)*_g(N-1));
  solution[N-2] = _g(N-1) - _g(N-2);//std::exp(-std::sqrt(-2*E)*_g(N-2));
  for (int i = N-2; i > 0; --i) {
    solution[i-1] = ((12 - f[i]*10)*solution[i] - f[i+1]*solution[i+1])/f[i-1];
  }
}
void HF::solveOutward(double E, int n, int l, std::vector<double> &solution, std::vector<double> &f) {
  int N = _g.N();
  solution[0] = std::pow(_Z*_g(0), l+0.5);
  solution[1] = std::pow(_Z*_g(1), l+0.5);
  for (int i = 0; i < N-1; ++i) {
    solution[i+1] = ((12 - f[i]*10)*solution[i] - f[i-1]*solution[i-1])/f[i+1];
  }
}
void HF::match(Orbital &o, int l, int m, int icl, std::vector<double> &inward, std::vector<double> &outward) {
  double ratio = outward[icl]/inward[icl];
  for (int i = 0; i < _g.N(); ++i) {
    if (i < icl) {
      o(i, l, m) = outward[i];
    } else {
      o(i, l, m) = ratio*inward[i];
    }
  }
}
long double HF::solveOrbitalFixedEnergy(double E, int k, int icl, int &nodes) {
  long double F = 0;
  nodes = 0;
  for (int l = 0; l < _o[k].L()+1; ++l) {
    for (int m = -l; m < l+1; ++m) {

      // TODO loop here to solve system
      // calculate auxiliary function f for Numerov method
      std::vector<double> f(_g.N(), 0);
      for (int i = 0; i < _g.N(); ++i) {
        double r = _g(i);
        double a = 2*std::pow(r, 2)*(E - _pot[i] - _vd[k][std::pair<int, int>(l, m)][i]) - std::pow(l + 0.5, 2);
        f[i] = 1 + a*std::pow(_g.dx(), 2)/12.0;
      }

      std::vector<double> inward(_g.N(), 0);
      std::vector<double> outward(_g.N(), 0);
      solveInward(E, _o[k].initialN(), l, inward, f);
      solveOutward(E, _o[k].initialN(), l, outward, f);
      match(_o[k], l, m, icl, inward, outward);

      int this_nodes = 0;
      for (int i = 3; i < _g.N()-3; ++i) {
        if (_o[k](i, l, m)*_o[k](i+1, l, m) < 0) this_nodes += 1;
      }
      nodes += this_nodes;

      if (m == _o[k].initialM() && l == _o[k].initialL())
        F += std::pow((12 - 10*f[icl])*_o[k](icl, l, m) - f[icl-1]*_o[k](icl-1, l, m) - f[icl+1]*_o[k](icl+1, l, m), 2);
    }
  }

  return F;
}

void HF::addOrbital(int L, int initial_n, int initial_l, int initial_m) {
  _o.push_back(Orbital(_g.N(), L, initial_n, initial_l, initial_m));
  // initialise energies
  for (int k = 0; k < _o.size(); ++k) {
    _o[k].E(-_Z*_Z*0.5/std::pow(_o[k].initialN(), 2));
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        double v = 0;
        if (l == _o[k].initialL() && m == _o[k].initialM()) v = 1;
        for (int ir = 0; ir < _g.N(); ++ir) { // for each radial point
          _o[k](ir, l, m) = v;
        }
      }
    }
  }
  _vd.clear();
  _vex.clear();
  for (int k = 0; k < _o.size(); ++k) {
    _vd[k] = Vd();
    _vd[k][std::pair<int, int>(0, 0)] = std::vector<double>(_g.N(), 0);
    _vd[k][std::pair<int, int>(1, -1)] = std::vector<double>(_g.N(), 0);
    _vd[k][std::pair<int, int>(1, 0)] = std::vector<double>(_g.N(), 0);
    _vd[k][std::pair<int, int>(1, 1)] = std::vector<double>(_g.N(), 0);
    _vex[k] = Vex();
    _vex[k][std::pair<int, int>(0, 0)] = std::vector<double>(_g.N(), 0);
    _vex[k][std::pair<int, int>(1, -1)] = std::vector<double>(_g.N(), 0);
    _vex[k][std::pair<int, int>(1, 0)] = std::vector<double>(_g.N(), 0);
    _vex[k][std::pair<int, int>(1, 1)] = std::vector<double>(_g.N(), 0);
  }
}


