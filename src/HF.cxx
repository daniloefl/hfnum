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

#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>

#include <Python.h>
using namespace boost;

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>

HF::HF()
  : SCF() {
}

HF::HF(const std::string fname)
  : SCF(fname) {
}

HF::HF(Grid &g, ldouble Z)
  : SCF(g, Z) {
}

HF::HF(python::object o, ldouble Z)
  : SCF(o, Z) {
}

HF::~HF() {
}

ldouble HF::getE0() {
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    E0 += _o[k]->E();
  }
  ldouble J = 0;
  ldouble K = 0;
  for (auto &vditm : _vd) {
    int k = vditm.first;
    int l = _o[k]->initialL();
    int m = _o[k]->initialM();
    for (int ir = 0; ir < _g->N(); ++ir) {
      ldouble r = (*_g)(ir);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      J += _vd[k][std::pair<int,int>(l, m)][ir]*std::pow(_o[k]->getNorm(ir, l, m, *_g), 2)*std::pow(r, 2)*dr;
    }
  }
  for (auto &vexitm : _vex) {
    const int k1 = vexitm.first.first;
    const int k2 = vexitm.first.second;
    int l1 = _o[k1]->initialL();
    int m1 = _o[k1]->initialM();
    int l2 = _o[k2]->initialL();
    int m2 = _o[k2]->initialM();
    for (int ir = 0; ir < _g->N(); ++ir) {
      ldouble r = (*_g)(ir);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      K += _vex[std::pair<int,int>(k1, k2)][std::pair<int,int>(l2, m2)][ir]*_o[k1]->getNorm(ir, l1, m1, *_g)*_o[k2]->getNorm(ir, l2, m2, *_g)*std::pow(r, 2)*dr;
    }
  }
  E0 += -0.5*(J - K);
  return E0;
}

void HF::solve(int NiterSCF, int Niter, ldouble F0stop) {
  _dE.resize(_o.size());
  _nodes.resize(_o.size());
  _Emax.resize(_o.size());
  _Emin.resize(_o.size());
  icl.resize(_o.size());


  int nStepSCF = 0;
  while (nStepSCF < NiterSCF) {
    for (int k = 0; k < _o.size(); ++k) {
      icl[k] = -1;

      ldouble lmain_eq = _o[k]->initialL();
      int lmain = _o[k]->initialL();
      int mmain = _o[k]->initialM();
      // calculate crossing of potential at zero for lmain,mmain
      ldouble a_m1 = 0;
      for (int i = 3; i < _g->N()-3; ++i) {
        ldouble r = (*_g)(i);
        ldouble a = 0;
        //if (_g.isLog()) a = 2*std::pow(r, 2)*(_o[k]->E() - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i]) - std::pow(lmain_eq + 0.5, 2);
        //else a = 2*(_o[k]->E() - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i]) - lmain_eq*(lmain_eq+1)/std::pow(r, 2);
        if (_g->isLog()) a = 2*std::pow(r, 2)*(_o[k]->E() - _pot[i]) - std::pow(lmain_eq + 0.5, 2);
        else a = 2*(_o[k]->E() - _pot[i]) - lmain_eq*(lmain_eq+1)/std::pow(r, 2);
        if (icl[k] < 0 && a*a_m1 < 0) {
          icl[k] = i;
          break;
        }
        a_m1 = a;
      }
      if (icl[k] < 0) icl[k] = 10;
      std::cout << "Found classical crossing for orbital " << k << " at " << icl[k] << std::endl;
    }

    for (int k = 0; k < _o.size(); ++k) {
      _nodes[k] = 0;
      _Emin[k] = -_Z*_Z;
      _Emax[k] = 0;
      //_o[k]->E(-_Z*_Z*0.5/std::pow(_o[k]->initialN(), 2));
    }

    std::cout << "SCF step " << nStepSCF << std::endl;
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
  std::cout << "Calculating Vex." << std::endl;

  for (int k = 0; k < _o.size(); ++k) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vexsum[std::pair<int, int>(k, k2)] = Vex();
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
    }
  }

  std::vector<int> avg;
  std::vector<int> done;
  std::vector<int> not_done;

  if (_central) {
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      avg.push_back(k1);
    }
  } else {
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      int nSameShell = 0;
      for (int kx = 0; kx < _o.size(); ++kx) {
        if (_o[k1]->initialN() == _o[kx]->initialN() && _o[k1]->initialL() == _o[kx]->initialL())
          nSameShell++;
      }
      if (nSameShell == 2*(2*_o[k1]->initialL() + 1)) {
        done.push_back(k1);
      } else {
        not_done.push_back(k1);
      }
    }
  }

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->initialL();
    int m1 = _o[k1]->initialM();

    // calculate it first with filled orbitals, dividing by the number of orbitals
    // this is exact if all 2(2*l+1) orbitals in this level are filled
    for (auto k2 : avg) {
      //if (k1 == k2) continue;
      if (_o[k1]->spin()*_o[k2]->spin() < 0) continue;

      int l2 = _o[k2]->initialL();
      int m2 = _o[k2]->initialM();
      std::cout << "Calculating Vex term from k1 = " << k1 << ", k2 = " << k2 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

      // temporary variable
      std::vector<ldouble> vex(_g->N(), 0); // calculate it here first
      for (int L = (int) std::fabs(l1 - l2); L <= l1 + l2; ++L) {
        ldouble coeff = 1.0/((ldouble) (2*L + 1))*std::pow(CG(l1, l2, 0, 0, L, 0), 2);
        for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
          ldouble r1 = (*_g)(ir1);
          ldouble rmax = r1;
          for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
            ldouble r2 = (*_g)(ir2);
            ldouble rmin = r2;

            ldouble dr = 0;
            if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);
            if (ir2 > ir1) {
              rmax = r2;
              rmin = r1;
            } else {
              rmax = r1;
              rmin = r2;
            }
            vex[ir1] += coeff*_o[k1]->getNorm(ir2, l1, m1, *_g)*_o[k2]->getNorm(ir2, l2, m2, *_g)*std::pow(r2, 2)*std::pow(rmin, L)/std::pow(rmax, L+1)*dr;
          }
        }
      }

      for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
        _vexsum[std::pair<int,int>(k1, k2)][std::pair<int,int>(l2, m2)][ir1] += vex[ir1];
      }
    }
  }

  // we are looking to calculate Vex(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_ko(r1) Y_k1(O)*Y_ko(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    int lj = _o[ko]->initialL();
    int mj = _o[ko]->initialM();

    // calculate it first with filled orbitals
    // loop over orbitals (this is the sum over k1 above)
    for (auto k1 : done) {
      if (ko == k1) continue;
      if (_o[k1]->spin()*_o[ko]->spin() < 0) continue;

      std::cout << "Calculating Vex for orbital eq. " << ko << ", term from k1 = " << k1 << " (assuming central potential)" << std::endl;

      int l1 = _o[k1]->initialL();
      int m1 = _o[k1]->initialM();

      std::vector<ldouble> vex(_g->N(), 0); // calculate it here first

      ldouble Q = 0;
      std::vector<ldouble> E(_g->N(), 0); // electric field
      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        ldouble r2 = (*_g)(ir2);
        ldouble dr = 0;
        if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);
        Q += (_o[k1]->getNorm(ir2, l1, m1, *_g)*_o[ko]->getNorm(ir2, lj, mj, *_g))*std::pow(r2, 2)*dr;
        E[ir2] = Q/std::pow(r2, 2);
      }
      vex[_g->N()-1] = Q/(*_g)(_g->N()-1);
      for (int ir2 = _g->N()-2; ir2 >= 0; --ir2) {
        ldouble dr = 0;
        if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);
        vex[ir2] = vex[ir2+1] + E[ir2]*dr;
      }

      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        _vexsum[std::pair<int,int>(ko, k1)][std::pair<int,int>(l1, m1)][ir2] += vex[ir2];
      }
    }
  }

  // for partially filled orbitals the angular dependence does not cancel out
  // need to expand all terms in spherical harmonics and integrate it
  for (int ko = 0; ko < _o.size(); ++ko) {
    for (int idx = 0; idx < _o[ko]->getSphHarm().size(); ++idx) {
      int lo = _o[ko]->getSphHarm()[idx].first;
      int mo = _o[ko]->getSphHarm()[idx].second;
      for (auto k1 : not_done) {
        std::cout << "Calculating Vex for orbital eq. " << ko << ", term from k1 = " << k1 << " (non-central potential)" << std::endl;

        std::vector<ldouble> &vexsum_curr = _vexsum[std::pair<int,int>(ko,k1)][std::pair<int,int>(lo, mo)];

        if (_o[k1]->spin()*_o[ko]->spin() < 0) continue;

        // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
        for (int idx1 = 0; idx1 < _o[k1]->getSphHarm().size(); ++idx1) {
          int l1 = _o[k1]->getSphHarm()[idx1].first;
          int m1 = _o[k1]->getSphHarm()[idx1].second;

          // now actually calculate it from the expansion above
          int lmax = 2;
          for (int l = 0; l < lmax+1; ++l) {
            for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
              ldouble beta = 0;
              ldouble r2 = (*_g)(ir2);
              for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
                ldouble r1 = (*_g)(ir1);
                ldouble dr = 0;
                if (ir1 < _g->N()-1) dr = (*_g)(ir1+1) - (*_g)(ir1);
                ldouble rs = r1;
                ldouble rb = r2;
                if (rb < rs) {
                  rs = r2;
                  rb = r1;
                }
                beta += 4*M_PI/(2.0*l + 1.0)*_o[k1]->getNorm(ir1, l1, m1, *_g)*_o[ko]->getNorm(ir1, lo, mo, *_g)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
              }
              ldouble T = 0;
              for (int m = -l; m < l + 1; ++m) {
                ldouble T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                // int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^mo int Ylm Ylo(-mo) Yljmj dOb = (-1)^(mo+mj) sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
                ldouble T2 = 0;
                T2 = std::sqrt((2*l+1)/(4*M_PI))*CG(l, lo, 0, 0, lo, 0)*CG(l, lo, m, -mo, lo, -mo);
                T += T1*T2;
              }
              vexsum_curr[ir2] += beta*T;
            } // for each r in Vex integration
          } // for each l in the 1/|r1 - r2| expansion in sph. harm.

        } // for each orbital in the Vex sum

      } // for each Vd term in a unique equation coming from the multiplication by Y*_i
    } // for each (l,m) in the orbital basis expansion
  }

  for (int ko = 0; ko < _o.size(); ++ko) {
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      for (auto &idx : _vex[std::pair<int,int>(ko,k1)]) {
        int lj = idx.first.first;
        int mj = idx.first.second;
        std::vector<ldouble> &currentVex = _vex[std::pair<int,int>(ko,k1)][std::pair<int,int>(lj, mj)];
        for (int k = 0; k < _g->N(); ++k) currentVex[k] = (1-gamma)*currentVex[k] + gamma*_vexsum[std::pair<int,int>(ko,k1)][std::pair<int,int>(lj,mj)][k];
      }
    }
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
  std::cout << "Calculating Vd." << std::endl;

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vd();
    _vdsum[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
  }

  std::vector<int> avg;
  std::vector<int> done;
  std::vector<int> not_done;

  if (_central) {
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      avg.push_back(k1);
    }
  } else {
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      int nSameShell = 0;
      for (int kx = 0; kx < _o.size(); ++kx) {
        if (_o[k1]->initialN() == _o[kx]->initialN() && _o[k1]->initialL() == _o[kx]->initialL())
          nSameShell++;
      }
      if (nSameShell == 2*(2*_o[k1]->initialL() + 1)) {
        done.push_back(k1);
      } else {
        not_done.push_back(k1);
      }
    }
  }

  std::map<int, std::map<std::pair<int,int>, std::vector<ldouble> > > vdsum; // calculate it here first

  // calculate it first with filled orbitals, dividing by the number of orbitals
  // this is exact if all 2(2*l+1) orbitals in this level are filled
  for (auto k1 : avg) {
    int l1 = _o[k1]->initialL();
    int m1 = _o[k1]->initialM();
    std::cout << "Calculating Vd term from k = " << k1 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

    // temporary variable
    std::vector<ldouble> vd(_g->N(), 0); // calculate it here first
    for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
      ldouble r1 = (*_g)(ir1);
      ldouble rmax = r1;
      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        ldouble r2 = (*_g)(ir2);
        ldouble dr = 0;
        if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);
        if (ir2 > ir1) rmax = r2;
        else rmax = r1;
        vd[ir1] += std::pow(_o[k1]->getNorm(ir2, l1, m1, *_g), 2)*std::pow(r2, 2)/rmax*dr;
      }
    }

    for (int ko = 0; ko < _o.size(); ++ko) {
      //if (ko == k1) continue;
      int lj = _o[ko]->initialL();
      int mj = _o[ko]->initialM();
      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        _vdsum[ko][std::pair<int,int>(lj,mj)][ir2] += vd[ir2];
      }
    }
  }

  // calculate it first with filled orbitals
  // loop over orbitals (this is the sum over k1 above)
  for (auto k1 : done) {
    int l1 = _o[k1]->initialL();
    int m1 = _o[k1]->initialM();
    std::cout << "Calculating Vd term from k1 = " << k1 << " (assuming central potential)" << std::endl;

    // temporary variable
    std::vector<ldouble> vd(_g->N(), 0); // calculate it here first

    ldouble Q = 0;
    std::vector<ldouble> E(_g->N(), 0); // electric field
    for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
      ldouble r2 = (*_g)(ir2);
      ldouble dr = 0;
      if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);
      Q += std::pow(_o[k1]->getNorm(ir2, l1, m1, *_g), 2)*std::pow(r2, 2)*dr;
      E[ir2] = Q/std::pow(r2, 2);
    }
    vd[_g->N()-1] = Q/(*_g)(_g->N()-1);
    for (int ir2 = _g->N()-2; ir2 >= 0; --ir2) {
      ldouble dr = (*_g)(ir2+1) - (*_g)(ir2);
      vd[ir2] = vd[ir2+1] + E[ir2]*dr;
    }

    for (int ko = 0; ko < _o.size(); ++ko) {
      if (ko == k1) continue;
      int lj = _o[ko]->initialL();
      int mj = _o[ko]->initialM();
      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        _vdsum[ko][std::pair<int,int>(lj,mj)][ir2] += vd[ir2];
      }
    }
  }

  // we are looking to calculate Vd(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_k1(r1) Y_k1(O)*Y_k1(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
    for (int idx = 0; idx < _o[ko]->getSphHarm().size(); ++idx) {
      int lo = _o[ko]->getSphHarm()[idx].first;
      int mo = _o[ko]->getSphHarm()[idx].second;

      // now with open shells
      for (auto k1 : not_done) {
        std::cout << "Calculating Vd term for eq. " << ko << ", due to k1 = " << k1 << " (non-central potential)" << std::endl;
        for (int idx1 = 0; idx1 < _o[k1]->getSphHarm().size(); ++idx1) {
          int l1 = _o[k1]->getSphHarm()[idx1].first;
          int m1 = _o[k1]->getSphHarm()[idx1].second;

          // now actually calculate it from the expansion above
          int lmax = 2;
          for (int l = 0; l < lmax+1; ++l) {
            for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
              ldouble beta = 0;
              ldouble r2 = (*_g)(ir2);
              for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
                ldouble r1 = (*_g)(ir1);
                ldouble dr = 0;
                if (ir1 < _g->N()-1) dr = (*_g)(ir1+1) - (*_g)(ir1);
                ldouble rs = r1;
                ldouble rb = r2;
                if (rb < rs) {
                  rs = r2;
                  rb = r1;
                }
                beta += 4*M_PI/(2.0*l + 1.0)*std::pow(_o[k1]->getNorm(ir1, l1, m1, *_g), 2)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
              }
              ldouble T = 0;
              for (int m = -l; m < l + 1; ++m) {
                ldouble T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                // int Ylm Y*lomo(Ob) Yljmj(Ob) dOb = (-1)^mo int Ylm Ylo(-mo) Yljmj dOb = (-1)^(mo+mj) sqrt((2l+1)*(2lo+1)/(4pi*(2lj+1))) * CG(l, lo, 0, 0, lj, 0) * CG(l, lo, m, -mo, lj, -mj)
                ldouble T2 = 0;
                T2 = std::sqrt((2*l+1)/(4*M_PI))*CG(l, lo, 0, 0, lo, 0)*CG(l, lo, m, -mo, lo, -mo);
                //if (m == 0 && l == 0) T2 = std::pow(4*M_PI, -0.5);
                T += T1*T2;
              }
              _vdsum[ko][std::pair<int,int>(lo,mo)][ir2] += beta*T;
            } // for each r in Vd integration
          } // for each l in the 1/|r1 - r2| expansion in sph. harm.
        } // for each orbital in the Vd sum
      }
    } // for each (l,m) in the orbital basis expansion
  }

  for (int ko = 0; ko < _o.size(); ++ko) {
    for (auto &idx : _vd[ko]) {
      int lj = idx.first.first;
      int mj = idx.first.second;
      std::cout << "Adding Vd term for eq. " << ko << std::endl;
      std::vector<ldouble> &currentVd = _vd[ko][std::pair<int,int>(lj, mj)];
      for (int k = 0; k < _g->N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*_vdsum[ko][std::pair<int,int>(lj,mj)][k];
    }
  }
}


void HF::calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E) {
  std::vector<MatrixXld> Lambda(_g->N());
  int N = _om.N();
  F.resize(_g->N());
  K.resize(_g->N());

  for (int i = 0; i < _g->N(); ++i) {
    ldouble r = (*_g)(i);
    F[i].resize(N, N);
    F[i].setZero();
    Lambda[i].resize(N, N);
    Lambda[i].setZero();
    K[i].resize(N, N);
    K[i].setZero();

    for (int idx1 = 0; idx1 < N; ++idx1) {
      int k1 = _om.orbital(idx1);
      int l1 = _om.l(idx1);
      ldouble l1_eq = _om.l(idx1);
      int m1 = _om.m(idx1);

      for (int idx2 = 0; idx2 < N; ++idx2) {
        int k2 = _om.orbital(idx2);
        int l2 = _om.l(idx2);
        ldouble l2_eq = _om.l(idx2);
        int m2 = _om.m(idx2);

        if (idx1 == idx2) {
          ldouble a = 0;
          if (_g->isLog()) a = 2*std::pow(r, 2)*(E[k1] - _pot[i] - _vd[k1][std::pair<int, int>(l1, m1)][i]) - std::pow(l1_eq + 0.5, 2);
          else a = 2*(E[k1] - _pot[i] - _vd[k1][std::pair<int, int>(l1, m1)][i] - l1_eq*(l1_eq + 1)/std::pow((*_g)(i), 2));

          F[i](idx1,idx1) += 1 + a*std::pow(_g->dx(), 2)/12.0;
          Lambda[i](idx1,idx1) += 1 + a*std::pow(_g->dx(), 2)/12.0;
        }
        ldouble vex = _vex[std::pair<int,int>(k1, k2)][std::pair<int, int>(l2, m2)][i];
        ldouble a = 0;

        if (_g->isLog()) a = 2*std::pow(r, 2)*vex;
        else a = 2*vex;

        F[i](idx1,idx2) += a*std::pow(_g->dx(), 2)/12.0;
        if (idx1 != idx2) K[i](idx1, idx2) += a;
        else Lambda[i](idx1, idx2) += a*std::pow(_g->dx(), 2)/12.0;
      }
    }
    K[i] = F[i].inverse();
    //for (int idxD = 0; idxD < N; ++idxD) Lambda[i](idxD, idxD) = 1.0/Lambda[i](idxD, idxD);
    //K[i] = Lambda[i]*K[i];
    //K[i] = (MatrixXld::Identity(N,N) + std::pow(_g->dx(), 2)/12.0*K[i] + std::pow(_g->dx(), 4)/144.0*(K[i]*K[i]))*Lambda[i];
  }
}

