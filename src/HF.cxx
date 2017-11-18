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

HF::HF(const Grid &g, ldouble Z)
  : _g(g), _Z(Z) {
  _pot.resize(_g.N());
  _potIndep.resize(_g.N());
  for (int k = 0; k < _g.N(); ++k) {
    _pot[k] = -_Z/_g(k);
    _potIndep[k] = 0;
  }
  _gamma_scf = 0.5;
  _norm = 1;
}

HF::~HF() {
}

std::vector<ldouble> HF::getOrbital(int no, int lo, int mo) {
  Orbital &o = _o[no];
  std::vector<ldouble> res;
  for (int k = 0; k < _g.N(); ++k) {
    res.push_back(o.getNorm(k, lo, mo, _g));
  }
  return res;
}

void HF::gammaSCF(ldouble g) {
  _gamma_scf = g;
}

void HF::solve(int NiterSCF, int Niter, ldouble F0stop) {
  _dE.resize(_o.size());
  int nStepSCF = 0;
  while (nStepSCF < NiterSCF) {
    std::cout << "SCF step " << nStepSCF << std::endl;
    //if (nStepSCF == 1) {
    //  _o[0].E(-3.301485557103038);
    //  _o[1].E(-3.325613315360123);
    //  _o[2].E(-0.6090752422026833);
    //}
    solveForFixedPotentials(Niter, F0stop);
    nStepSCF++;
    calculateVex(_gamma_scf);
    calculateVd(_gamma_scf);
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
void HF::calculateVex(ldouble gamma) {
  // we are looking to calculate Vex(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_ko(r1) Y_k1(O)*Y_ko(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
    for (int lj = 0; lj < _o[ko].L()+1; ++lj) { // loop over l in Y_j
      for (int mj = -lj; mj < lj+1; ++mj) { // loop over m in Y_j

        for (int k1 = 0; k1 < _o.size(); ++k1) {
          //if (k1 == ko) continue;
          // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
          for (int l1 = 0; l1 < _o[k1].L()+1; ++l1) { // loop over l in Y_j
            for (int m1 = -l1; m1 < l1+1; ++m1) { // loop over m in Y_j

              for (auto &vexLm : _vex[std::pair<int,int>(ko,k1)]) { // calculate a term in square brackets above
                int vexl = vexLm.first.first; // these are the l and m for Y*_i
                int vexm = vexLm.first.second;
                if (vexl != lj || vexm != mj) continue;
                std::vector<ldouble> &currentVex = vexLm.second; // this is the r-dependent part

                std::vector<ldouble> vex(_g.N(), 0); // calculate it here first

                // loop over orbitals (this is the sum over k1 above)
                if (_o[k1].spin() != _o[ko].spin()) continue;

                //int nSameShell = 0;
                //for (int kx = 0; kx < _o.size(); ++kx) {
                //  if (_o[k1].initialN() == _o[kx].initialN() && _o[k1].initialL() == _o[kx].initialL()) nSameShell += 1;
                //}
                //if (nSameShell == 2*_o[ko].initialL() + 1) { // filled sub-shell
                //  for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
                //    ldouble r2 = _g(ir2);
                //    for (int l = abs(l1-lj); l < l1+lj+1; ++l) {
                //      ldouble beta = 0;
                //      for (int ir1 = 0; ir1 < _g.N(); ++ir1) {
                //        ldouble r1 = _g(ir1);
                //        ldouble dr = 0;
                //        if (ir1 < _g.N()-1) dr = _g(ir1+1) - _g(ir1);
                //        ldouble rs = r1;
                //        ldouble rb = r2;
                //        if (rb < rs) {
                //          rs = r2;
                //          rb = r1;
                //        }
                //        beta += _o[k1].getNorm(ir1, l1, m1, _g)*_o[ko].getNorm(ir1, lj, mj, _g)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
                //      }
                //      vex[ir2] += 1.0/((double) nSameShell)*((double) nSameShell)/(2.0*l + 1.0)*std::pow(CG(l1, lj, 0, 0, l, 0), 2)*beta;
                //    }
                //  }
                //} else { // not filled sub-shell, need to sum over l and m in the spherical harmonics expansion

                  // now actually calculate it from the expansion above
                  int lmax = 2;
                  for (int l = 0; l < lmax+1; ++l) {
                    for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
                      ldouble beta = 0;
                      ldouble r2 = _g(ir2);
                      for (int ir1 = 0; ir1 < _g.N(); ++ir1) {
                        ldouble r1 = _g(ir1);
                        ldouble dr = 0;
                        if (ir1 < _g.N()-1) dr = _g(ir1+1) - _g(ir1);
                        ldouble rs = r1;
                        ldouble rb = r2;
                        if (rb < rs) {
                          rs = r2;
                          rb = r1;
                        }
                        beta += 4*M_PI/(2.0*l + 1.0)*_o[k1].getNorm(ir1, l1, m1, _g)*_o[ko].getNorm(ir1, lj, mj, _g)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
                      }
                      ldouble T = 0;
                      for (int m = -l; m < l + 1; ++m) {
                        ldouble T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                        // int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^mo int Ylm Ylo(-mo) Yljmj dOb = (-1)^(mo+mj) sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
                        ldouble T2 = 0;
                        //T2 = std::pow(-1, vexm+mj)*std::sqrt((2*l+1)*(2*vexl+1)/(4*M_PI*(2*lj+1)))*CG(l, vexl, 0, 0, lj, 0)*CG(l, vexl, m, -vexm, lj, -mj);
                        if (m == 0 && l == 0) T2 = std::pow(4*M_PI, -0.5);
                        T += T1*T2;
                      }
                      vex[ir2] += beta*T;
                    } // for each r in Vex integration
                  } // for each l in the 1/|r1 - r2| expansion in sph. harm.
                //} // if sub-shell is filled
                for (int k = 0; k < _g.N(); ++k) currentVex[k] = (1-gamma)*currentVex[k] + gamma*vex[k];
              } // for each m1 of the orbital basis expansion

            } // for each l1 of the orbital basis expansion

          } // for each orbital in the Vex sum

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
void HF::calculateVd(ldouble gamma) {
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
          std::vector<ldouble> &currentVd = vdLm.second; // this is the r-dependent part

          std::vector<ldouble> vd(_g.N(), 0); // calculate it here first
          // loop over orbitals (this is the sum over k1 above)
          for (int k1 = 0; k1 < _o.size(); ++k1) {
            //int nSameShell = 0;
            //for (int kx = 0; kx < _o.size(); ++kx) {
            //  if (_o[k1].initialN() == _o[kx].initialN() && _o[k1].initialL() == _o[kx].initialL()) nSameShell += 1;
            //}
            //if (nSameShell == 2*_o[ko].initialL() + 1) { // filled sub-shell
            //  for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
            //    ldouble beta = 0;
            //    ldouble r2 = _g(ir2);
            //    for (int ir1 = 0; ir1 < _g.N(); ++ir1) {
            //      ldouble r1 = _g(ir1);
            //      ldouble dr = 0;
            //      if (ir1 < _g.N()-1) dr = _g(ir1+1) - _g(ir1);
            //      ldouble rs = r1;
            //      ldouble rb = r2;
            //      if (rb < rs) {
            //        rs = r2;
            //        rb = r1;
            //      }
            //      beta += 1.0/((double) nSameShell)*(2.0*_o[k1].initialL() + 1.0)*std::pow(_o[k1].getNorm(ir1, lj, mj, _g), 2)/rb*std::pow(r1, 2)*dr;
            //    }
            //    vd[ir2] += beta;
            //  }
            //} else { // not filled sub-shell, need to sum over l and m in the spherical harmonics expansion
              for (int l1 = 0; l1 < _o[k1].L()+1; ++l1) { // each orbital in the sum is actually orb_ko = sum_l1,m1 rpsi_l1,m1 Y_l1m1, so loop over this sum
                for (int m1 = -l1; m1 < l1+1; ++m1) {

                  // now actually calculate it from the expansion above
                  int lmax = 2;
                  for (int l = 0; l < lmax+1; ++l) {
                    for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
                      ldouble beta = 0;
                      ldouble r2 = _g(ir2);
                      for (int ir1 = 0; ir1 < _g.N(); ++ir1) {
                        ldouble r1 = _g(ir1);
                        ldouble dr = 0;
                        if (ir1 < _g.N()-1) dr = _g(ir1+1) - _g(ir1);
                        ldouble rs = r1;
                        ldouble rb = r2;
                        if (rb < rs) {
                          rs = r2;
                          rb = r1;
                        }
                        beta += 4*M_PI/(2.0*l + 1.0)*std::pow(_o[k1].getNorm(ir1, l1, m1, _g), 2)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
                      }
                      ldouble T = 0;
                      for (int m = -l; m < l + 1; ++m) {
                        ldouble T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                        // int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^mo int Ylm Ylo(-mo) Yljmj dOb = (-1)^(mo+mj) sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
                        ldouble T2 = 0;
                        T2 = std::pow(-1, vdm+mj)*std::sqrt((2*l+1)*(2*vdl+1)/(4*M_PI*(2*lj+1)))*CG(l, vdl, 0, 0, lj, 0)*CG(l, vdl, m, -vdm, lj, -mj);
                        T += T1*T2;
                      }
                      vd[ir2] += beta*T;
                    } // for each r in Vd integration
                  } // for each l in the 1/|r1 - r2| expansion in sph. harm.
                } // for each m1 of the orbital basis expansion
              } // for each l1 of the orbital basis expansion

            //} // if sub-shell filled
          } // for each orbital in the Vd sum

          for (int k = 0; k < _g.N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*vd[k];

        } // for each Vd term in a unique equation coming from the multiplication by Y*_i
      } // for each m in the orbital basis expansion
    } // for each l in the orbital basis expansion

  }
}

std::vector<ldouble> HF::getNucleusPotential() {
  return _pot;
}

std::vector<ldouble> HF::getDirectPotential(int k) {
  return _vd[k][std::pair<int, int>(0, 0)];
}

std::vector<ldouble> HF::getExchangePotential(int k, int k2) {
  return _vex[std::pair<int,int>(k,k2)][std::pair<int, int>(0, 0)];
}

void HF::solveForFixedPotentials(int Niter, ldouble F0stop) {
  ldouble gamma = 0.5; // move in the direction of the negative slope with this velocity per step

  ldouble F = 0;
  int nStep = 0;
  while (nStep < Niter) {
    // compute sum of squares of F(x_old)
    nStep += 1;
    F = step();

    // limit maximum energy step in a single direction to be 0.1*gamma
    ldouble gscale = 1;

    // change orbital energies
    std::cout << "Orbital energies at step " << nStep << ", with constraint = " << std::setw(16) << F << "." << std::endl;
    std::cout << std::setw(5) << "Index" << " " << std::setw(16) << "Energy (H)" << " " << std::setw(16) << "next energy (H) " << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      ldouble stepdE = gscale*gamma*_dE[k];
      ldouble newE = (_o[k].E()+stepdE);
      std::cout << std::setw(5) << k << " " << std::setw(16) << std::setprecision(12) << _o[k].E() << " " << std::setw(16) << std::setprecision(12) << newE << std::endl;
      _o[k].E(newE);
    }

    if (std::fabs(F) < F0stop) break;
  }
}

void HF::calculateFMatrix(std::vector<MatrixXld> &F, std::vector<ldouble> &E) {
  int N = 0;
  for (int k = 0; k < _o.size(); ++k) {
    N += 2*_o[k].L()+1;
  }
  F.resize(_g.N());
  for (int i = 0; i < _g.N(); ++i) {
    ldouble r = _g(i);
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

                if (idx1 == idx2) {
                  ldouble a = 0;
                  if (_g.isLog()) a = 2*std::pow(r, 2)*(E[k] - _pot[i] - _vd[k][std::pair<int, int>(l, m)][i]) - std::pow(l + 0.5, 2);
                  else a = 2*(E[k] - _pot[i] - _vd[k][std::pair<int, int>(l, m)][i] - l*(l + 1)/std::pow(_g(i), 2));
                  F[i](idx1,idx1) += 1 + a*std::pow(_g.dx(), 2)/12.0;
                  //if (std::pow(a*_g.dx(), 2) > 6) {
                  //  std::cout << "Coefficient = " << a << " at i = " << i << ", r = " << r << " in (" << idx1 << ","<<idx1 <<"), which leads to (a*dx)^2 = " << std::pow(a*_g.dx(), 2) << " > 6 --> This will cause instabilities." << std::endl;
                  //}
                }
                ldouble vex = _vex[std::pair<int,int>(k, k2)][std::pair<int, int>(l2, m2)][i];
                ldouble a = 0;
                if (_g.isLog()) a = 2*std::pow(r, 2)*vex;
                else a = 2*vex;
                F[i](idx1,idx2) += a*std::pow(_g.dx(), 2)/12.0;
                //if (std::pow(a*_g.dx(), 2) > 6) {
                //  std::cout << "Coefficient = " << a << " at i = " << i << ", r = " << r << " in (" << idx1 << ","<<idx2 <<"), which leads to (a*dx)^2 = " << std::pow(a*_g.dx(), 2) << " > 6 --> This will cause instabilities." << std::endl;
                //}
                idx2 += 1;

              }
            }
          }
          idx1 += 1;
        }
      }
    }
  }
  //for (int k = 0; k < _o.size(); ++k) {
  //  std::cout << "Vd["<<k<<"] = " << _vd[k][std::pair<int, int>(0, 0)][2] << std::endl;
  //  for (int k2 = 0; k2 < _o.size(); ++k2) {
  //    std::cout << "Vex["<<k<<","<<k2<<"] = " << _vex[std::pair<int,int>(k,k2)][std::pair<int, int>(0, 0)][2] << std::endl;
  //  }
  //}
  //std::cout << "Calculate F at 2: " << F[2] << std::endl;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble HF::step() {
  // TODO: Ignore off-diagonal entries due to vxc now ... will add iteratively later
  // https://ocw.mit.edu/courses/mathematics/18-409-topics-in-theoretical-computer-science-an-algorithmists-toolkit-fall-2009/lecture-notes/MIT18_409F09_scribe21.pdf
  int N = 0;
  for (int k = 0; k < _o.size(); ++k) {
    N += _o[k].L()+1;
  }

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<int> icl(_o.size(), -1);
  for (int k = 0; k < _o.size(); ++k) {
    E[k] = _o[k].E();
    l[k] = _o[k].initialL();

    int lmain = _o[k].initialL();
    int mmain = _o[k].initialM();
    // calculate crossing of potential at zero for lmain,mmain
    ldouble a_m1 = 0;
    for (int i = 3; i < _g.N()-3; ++i) {
      ldouble r = _g(i);
      ldouble a = 0;
      if (_g.isLog()) a = 2*std::pow(r, 2)*(E[k] - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i] + _vex[std::pair<int,int>(k,k)][std::pair<int,int>(lmain, mmain)][i]) - std::pow(lmain + 0.5, 2);
      else a = 2*(E[k] - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i] + _vex[std::pair<int,int>(k,k)][std::pair<int,int>(lmain, mmain)][i] - lmain*(lmain+1)/std::pow(r, 2));
      if (icl[k] < 0 && a*a_m1 < 0) {
        icl[k] = i;
        break;
      }
      a_m1 = a;
    }
    if (icl[k] < 0) icl[k] = 10;
  }
  //for (int k = 0; k < _o.size(); ++k) {
  //  icl[k] = icl[_o.size()-1];
  //}

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = -1e-5;
  }

  std::vector<MatrixXld> Fmn;
  calculateFMatrix(Fmn, E);

  VectorXld Fn = solveOrbitalFixedEnergy(E, l, Fmn, icl);

  MatrixXld J;
  //J.resize(_o.size()+1, _o.size()+1);
  J.resize(_o.size(), _o.size());
  J.setZero();
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {

      std::vector<ldouble> EdE = E;
      EdE[k2] += dE[k2];

      std::vector<MatrixXld> Fmd;
      calculateFMatrix(Fmd, EdE);

      VectorXld Fd = solveOrbitalFixedEnergy(EdE, l, Fmd, icl);
      J(k1, k2) = (Fd[k1] - Fn[k1])/dE[k2];
    }
    //J(_o.size(),k1) += -2*E[k1];
  }
  //J(_o.size(),_o.size()) += 1;
  JacobiSVD<MatrixXld> decJ(J, ComputeThinU | ComputeThinV);
  VectorXld Fne;
  //Fne.resize(_o.size()+1, 1);
  Fne.resize(_o.size(), 1);
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    Fne(k1, 0) = Fn(k1, 0);
  }
  //Fne(_o.size(), 0) = 0;
  VectorXld dEv = -decJ.solve(Fne);

  ldouble F = 0;
  for (int k = 0; k < _o.size(); ++k) {
    _dE[k] = dEv(k);
    std::cout << "Orbital " << k << ", Fnominal = " << Fn[k] << ", dE(Jacobian) = " << dEv(k) << std::endl;
    F += Fn(k);
  }
  return F;
}

void HF::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm) {
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
          if (E[k] < 0) {
            solution[N-1](idx) = std::exp(-std::sqrt(-2*E[k])*_g(N-1));
            solution[N-2](idx) = std::exp(-std::sqrt(-2*E[k])*_g(N-2));
          } else {
            solution[N-1](idx) = 0;
            solution[N-2](idx) = (_g(N-1) - _g(N-2));
          }
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
          JacobiSVD<MatrixXld> dec(Fm[i-1], ComputeThinU | ComputeThinV);
          solution[i-1] = dec.solve((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1]));
        }
        idx += 1;
      }
    }
  }
}

void HF::solveOutward(std::vector<ldouble> &E, std::vector<int> &li, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm) {
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
          if (_g.isLog()) {
            solution[0](idx) = std::pow(_Z*_g(0)/((ldouble) _o[k].initialN()), li[k]+0.5);
            solution[1](idx) = std::pow(_Z*_g(1)/((ldouble) _o[k].initialN()), li[k]+0.5);
            if (_o[k].initialL() == 0) {
              solution[0](idx) = std::sqrt(_g(0))*2*std::exp(-_Z*_g(0)/((ldouble) _o[k].initialN()));
              solution[1](idx) = std::sqrt(_g(1))*2*std::exp(-_Z*_g(1)/((ldouble) _o[k].initialN()));
            }
          } else {
            solution[0](idx) = std::pow(_Z*_g(0)/((ldouble) _o[k].initialN()), li[k]+1);
            solution[1](idx) = std::pow(_Z*_g(1)/((ldouble) _o[k].initialN()), li[k]+1);
            if (_o[k].initialL() == 0) {
              solution[0](idx) = 2*std::exp(-_Z*_g(0)/((ldouble) _o[k].initialN()));
              solution[1](idx) = 2*std::exp(-_Z*_g(1)/((ldouble) _o[k].initialN()));
            }
          }
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
          JacobiSVD<MatrixXld> dec(Fm[i+1], ComputeThinU | ComputeThinV);
          solution[i+1] = dec.solve((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
        }
        idx += 1;
      }
    }
  }
}
void HF::match(std::vector<VectorXld> &o, std::vector<int> &icl, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward) {
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += _o[k].L()+1;
  }
  for (int i = 0; i < _g.N(); ++i) {
    o[i].resize(M);
  }

  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    ldouble ratio = outward[icl[k]](k)/inward[icl[k]](k);
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 0; i < _g.N(); ++i) {
          if (i <= icl[k]) {
            o[i](idx) = outward[i](idx);
          } else {
            o[i](idx) = ratio*inward[i](idx);
          }
        }
        idx += 1;
      }
    }
  }
}
VectorXld HF::solveOrbitalFixedEnergy(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<int> &icl) {
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += _o[k].L()+1;
  }

  VectorXld F;
  F.resize(_o.size());
  F.setZero();

  std::vector<VectorXld> inward(_g.N());
  std::vector<VectorXld> outward(_g.N());
  std::vector<VectorXld> matched(_g.N());

  solveOutward(E, l, outward, Fm);
  solveInward(E, l, inward, Fm);
  match(matched, icl, inward, outward);

  // calculate mis-match vector F
  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    //for (int l = 0; l < _o[k].L()+1; ++l) {
    //  for (int m = -l; m < l+1; ++m) {
    //    if (m == _o[k].initialM() && l == _o[k].initialL())
    //      F(k) += (12 - 10*Fm[icl[k]](idx))*matched[icl[k]](idx) - Fm[icl[k]-1](idx)*matched[icl[k]-1](idx) - Fm[icl[k]+1](idx)*matched[icl[k]+1](idx);
    //    idx += 1;
    //  }
    //}
    VectorXld prodIcl = (MatrixXld::Identity(M,M)*12 - Fm[icl[k]]*10)*matched[icl[k]] - Fm[icl[k]-1]*matched[icl[k]-1] - Fm[icl[k]+1]*matched[icl[k]+1];
    F(k) += prodIcl.transpose()*prodIcl;
    //for (int l = 0; l < _o[k].L()+1; ++l) {
    //  for (int m = -l; m < l+1; ++m) {
    //    if (m == _o[k].initialM() && l == _o[k].initialL())
    //      F(k) += prodIcl[idx];
    //    idx += 1;
    //  }
    //}
    //std::cout << "Orbital " << k << ": mismatch: " << F[k] << "in position " << _g(icl[k]) << "(" << icl[k] << ")" << " decomposition: " << prodIcl << std::endl;
  }
  
  idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 0; i < _g.N(); ++i) {
          _o[k](i, l, m) = matched[i](idx);
        }
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
        ldouble v = 0;
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
    _vd[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g.N(), 0);
    _vd[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g.N(), 0);
    _vd[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g.N(), 0);
    _vd[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g.N(), 0);
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vex[std::pair<int, int>(k, k2)] = Vex();
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g.N(), 0);
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g.N(), 0);
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g.N(), 0);
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g.N(), 0);
    }
  }
}


