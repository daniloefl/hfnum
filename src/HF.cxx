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

#include <Eigen/Core>
#include <Eigen/Dense>

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
    calculateVex(_gamma_scf);
  }
}

// exchange potential calculation
// V = int_Oa int_ra rpsi1(ra) rpsi2(ra) Yl1m1(Oa) Yl2m2(Oa)/|ra-rb| ra^2 dOa dra
// with 1 -> k1
// with 2 -> k2
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
void HF::calculateVex(double gamma) {
  // we are looking to calculate Vex(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_ko(r1) Y_k1(O)*Y_ko(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
    for (int lj = 0; lj < _o[ko].L()+1; ++lj) { // loop over l in Y_j
      for (int mj = -lj; mj < lj+1; ++mj) { // loop over m in Y_j

        for (auto &vexLm : _vex[ko]) { // calculate a term in square brackets above
          int vexl = vexLm.first.first; // these are the l and m for Y*_i
          int vexm = vexLm.first.second;
          if (vexl != lj || vexm != mj) continue;
          std::vector<double> &currentVex = vexLm.second; // this is the r-dependent part

          std::vector<double> vex(_g.N(), 0); // calculate it here first
          // loop over orbitals (this is the sum over k1 above)
          for (int k1 = 0; k1 < _o.size(); ++k1) {
            if (_o[k1].spin() != _o[ko].spin()) continue;

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
                      beta += 4*M_PI/(2.0*l + 1.0)*_o[k1].getNorm(ir1, l1, m1, _g)*_o[ko].getNorm(ir1, lj, mj, _g)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
                    }
                    double T = 0;
                    for (int m = -l; m < l + 1; ++m) {
                      double T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                      // int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^mo int Ylm Ylo(-mo) Yljmj dOb = (-1)^(mo+mj) sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
                      double T2 = 0;
                      T2 = std::pow(-1, vexm+mj)*std::sqrt((2*l+1)*(2*vexl+1)/(4*M_PI*(2*lj+1)))*CG(l, vexl, 0, 0, lj, 0)*CG(l, vexl, m, -vexm, lj, -mj);
                      T += T1*T2;
                    }
                    vex[ir2] += beta*T;
                  } // for each r in Vex integration
                } // for each l in the 1/|r1 - r2| expansion in sph. harm.
              } // for each m1 of the orbital basis expansion
            } // for each l1 of the orbital basis expansion

          } // for each orbital in the Vex sum

          for (int k = 0; k < _g.N(); ++k) currentVex[k] = (1-gamma)*currentVex[k] + gamma*vex[k];

        } // for each Vd term in a unique equation coming from the multiplication by Y*_i
      } // for each m in the orbital basis expansion
    } // for each l in the orbital basis expansion

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
          //for (int k = 0; k < _g.N(); ++k) vd[k] *= 0.5;

          for (int k = 0; k < _g.N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*vd[k];

        } // for each Vd term in a unique equation coming from the multiplication by Y*_i
      } // for each m in the orbital basis expansion
    } // for each l in the orbital basis expansion

  }
}

std::vector<double> HF::getNucleusPotential() {
  return _pot;
}

std::vector<double> HF::getDirectPotential(int k) {
  return _vd[k][std::pair<int, int>(0, 0)];
}

std::vector<double> HF::getExchangePotential(int k) {
  return _vex[k][std::pair<int, int>(0, 0)];
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

void HF::calculateFMatrix(std::vector<MatrixXd> &F, std::vector<double> &E) {
  int N = 0;
  for (int k = 0; k < _o.size(); ++k) {
    N += _o[k].L()+1;
  }
  F.resize(_g.N());
  for (int i = 0; i < _g.N(); ++i) {
    double r = _g(i);
    F[i].resize(N, N);
    F[i].setZero();
    int idx1 = 0;

    for (int k = 0; k < _o.size(); ++k) {
      for (int l = 0; l < _o[k].L()+1; ++l) {
        for (int m = -l; m < l+1; ++m) {

          int idx2 = 0;
          for (int k2 = 0; k2 < _o.size(); ++k2) {
            for (int l2 = 0; l2 < _o[k2].L()+1; ++l2) {
              for (int m2 = -l2; m2 < l2+1; ++m2) {
                double vex = _vex[k2][std::pair<int, int>(l2, m2)][i];
                if (_o[k2].spin() != _o[k].spin()) vex = 0;
                if (idx1 == idx2) {
                  double a = 2*std::pow(r, 2)*(E[k] - _pot[i] - _vd[k][std::pair<int, int>(l, m)][i] + vex) - std::pow(l + 0.5, 2);
                  F[i](idx1,idx1) += 1 + a*std::pow(_g.dx(), 2)/12.0;
                } else {
                  double a = 2*std::pow(r, 2)*vex;
                  F[i](idx1,idx2) += a*std::pow(_g.dx(), 2)/12.0;
                }
                idx2 += 1;
              }
            }
          }
          idx1 += 1;
        }
      }
    }
  }
}

// solve for a fixed energy and calculate _dE for the next step
long double HF::step() {
  // TODO: Ignore off-diagonal entries due to vxc now ... will add iteratively later
  // https://ocw.mit.edu/courses/mathematics/18-409-topics-in-theoretical-computer-science-an-algorithmists-toolkit-fall-2009/lecture-notes/MIT18_409F09_scribe21.pdf
  int N = 0;
  for (int k = 0; k < _o.size(); ++k) {
    N += _o[k].L()+1;
  }

  std::vector<double> E(_o.size(), 0);
  std::vector<double> EdE(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);
  double dE = 0.1;

  std::vector<int> icl(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    E[k] = _o[k].E();
    EdE[k] = E[k]+dE;
    l[k] = _o[k].initialL();

    int lmain = _o[k].initialL();
    int mmain = _o[k].initialM();
    // calculate crossing of potential at zero for lmain,mmain
    double a_m1 = 0;
    for (int i = 1; i < _g.N(); ++i) {
      double r = _g(i);
      double a = 2*std::pow(r, 2)*(E[k] - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i] + _vex[k][std::pair<int,int>(lmain, mmain)][i]) - std::pow(lmain + 0.5, 2);
      if (a*a_m1 < 0) {
        icl[k] = i;
        break;
      }
      a_m1 = a;
    }
  }

  long double F = 0;
  std::vector<MatrixXd> Fm1;
  std::vector<MatrixXd> Fm2;
  calculateFMatrix(Fm1, EdE);
  calculateFMatrix(Fm2, E);

  std::vector<double> F1 = solveOrbitalFixedEnergy(EdE, l, Fm1, icl);
  std::vector<double> F2 = solveOrbitalFixedEnergy(E, l, Fm2, icl);
  for (int k = 0; k < _o.size(); ++k) {
    if (F2[k] != F1[k]) {
      _dE[k] = -F2[k]*dE/(F1[k] - F2[k]);
    } else {
      _dE[k] = dE;
    }
    F += F2[k];
  }
  return F;
}

void HF::solveInward(std::vector<double> &E, std::vector<int> &l, std::vector<VectorXd> &solution, std::vector<MatrixXd> &Fm) {
  int N = _g.N();
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += _o[k].L()+1;
  }
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
  }
  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        if (l == _o[k].initialL() && m == _o[k].initialM()) {
          solution[N-1](idx) = 0;//std::exp(-std::sqrt(-2*E)*_g(N-1));
          solution[N-2](idx) = _g(N-1) - _g(N-2);//std::exp(-std::sqrt(-2*E)*_g(N-2));
        }
        idx += 1;
      }
    }
  }
  idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = N-2; i > 0; --i) {
          solution[i-1] = (Fm[i-1]).inverse()*((MatrixXd::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1]));
        }
        idx += 1;
      }
    }
  }
}

void HF::solveOutward(std::vector<double> &E, std::vector<int> &li, std::vector<VectorXd> &solution, std::vector<MatrixXd> &Fm) {
  int N = _g.N();
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += _o[k].L()+1;
  }
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
  }
  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        if (l == _o[k].initialL() && m == _o[k].initialM()) {
          solution[0](idx) = std::pow(_Z*_g(0), li[k]+0.5);
          solution[1](idx) = std::pow(_Z*_g(1), li[k]+0.5);
        }
        idx += 1;
      }
    }
  }
  idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 1; i < N-1; ++i) {
          solution[i+1] = (Fm[i+1]).inverse()*((MatrixXd::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
        }
        idx += 1;
      }
    }
  }
}
void HF::match(std::vector<VectorXd> &o, std::vector<int> &icl, std::vector<VectorXd> &inward, std::vector<VectorXd> &outward) {
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += _o[k].L()+1;
  }
  for (int i = 0; i < _g.N(); ++i) {
    o[i].resize(M);
  }

  for (int k = 0; k < _o.size(); ++k) {
    double ratio = outward[icl[k]](k)/inward[icl[k]](k);
    for (int i = 0; i < _g.N(); ++i) {
      if (i < icl[k]) {
        o[i](k) = outward[i](k);
      } else {
        o[i](k) = ratio*inward[i](k);
      }
    }
  }
}
std::vector<double> HF::solveOrbitalFixedEnergy(std::vector<double> &E, std::vector<int> &l, std::vector<MatrixXd> &Fm, std::vector<int> &icl) {
  std::vector<double> F(_o.size(), 0);

  std::vector<VectorXd> inward(_g.N());
  std::vector<VectorXd> outward(_g.N());
  std::vector<VectorXd> matched(_g.N());
  solveInward(E, l, inward, Fm);
  solveOutward(E, l, outward, Fm);
  match(matched, icl, inward, outward);

  // calculate mis-match vector F
  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 0; i < _g.N(); ++i) {
          _o[k](i, l, m) = matched[i](idx);
        }
        if (m == _o[k].initialM() && l == _o[k].initialL())
          F[k] += std::pow((12 - 10*Fm[icl[k]](idx, idx))*_o[k](icl[k], l, m) - Fm[icl[k]-1](idx, idx)*_o[k](icl[k]-1, l, m) - Fm[icl[k]+1](idx, idx)*_o[k](icl[k]+1, l, m), 2);
        idx += 1;
      }
    }
  }

  return F;
}

void HF::addOrbital(int L, int s, int initial_n, int initial_l, int initial_m) {
  _o.push_back(Orbital(_g.N(), s, L, initial_n, initial_l, initial_m));
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


