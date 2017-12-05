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
  : _g(g), _Z(Z), _om(_g, _o), _irs(_g, _o, icl, _om), _igs(_g, _o, icl, _om) {
  _pot.resize(_g.N());
  for (int k = 0; k < _g.N(); ++k) {
    _pot[k] = -_Z/_g(k);
  }
  _gamma_scf = 0.5;
  _method = 2;
}

HF::~HF() {
}

void HF::method(int m) {
  if (m < 0) m = 0;
  if (m > 2) m = 2;
  _method = m;
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
  _nodes.resize(_o.size());
  _Emax.resize(_o.size());
  _Emin.resize(_o.size());
  icl.resize(_o.size());


  int nStepSCF = 0;
  while (nStepSCF < NiterSCF) {
    std::cout << "Finding classical crossing." << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      icl[k] = -1;

      int lmain = _o[k].initialL();
      int mmain = _o[k].initialM();
      // calculate crossing of potential at zero for lmain,mmain
      ldouble a_m1 = 0;
      for (int i = 3; i < _g.N()-3; ++i) {
        ldouble r = _g(i);
        ldouble a = 0;
        if (_g.isLog()) a = 2*std::pow(r, 2)*(_o[k].E() - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i] + _vex[std::pair<int,int>(k,k)][std::pair<int,int>(lmain, mmain)][i]) - std::pow(lmain + 0.5, 2);
        else a = 2*(_o[k].E() - _pot[i] - _vd[k][std::pair<int, int>(lmain, mmain)][i] + _vex[std::pair<int,int>(k,k)][std::pair<int,int>(lmain, mmain)][i]) - lmain*(lmain+1)/std::pow(r, 2);
        if (icl[k] < 0 && a*a_m1 < 0) {
          icl[k] = i;
          break;
        }
        a_m1 = a;
      }
      if (icl[k] < 0) icl[k] = 10;
    }

    for (int k = 0; k < _o.size(); ++k) {
      _nodes[k] = 0;
      _Emin[k] = -_Z*_Z;
      _Emax[k] = 0;
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
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g.N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g.N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g.N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g.N(), 0);
    }
  }

  std::vector<int> done;
  std::vector<int> not_done;

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int nSameShell = 0;
    for (int kx = 0; kx < _o.size(); ++kx) {
      if (_o[k1].initialN() == _o[kx].initialN() && _o[k1].initialL() == _o[kx].initialL())
        nSameShell++;
    }
    if (nSameShell == 2*(2*_o[k1].initialL() + 1)) {
      done.push_back(k1);
    } else {
      done.push_back(k1);
      //not_done.push_back(k1);
    }
  }

  // we are looking to calculate Vex(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_ko(r1) Y_k1(O)*Y_ko(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    int lj = _o[ko].initialL();
    int mj = _o[ko].initialM();

    // calculate it first with filled orbitals
    // loop over orbitals (this is the sum over k1 above)
    for (auto k1 : done) {
      if (ko == k1) continue;
      if (_o[k1].spin()*_o[ko].spin() < 0) continue;

      std::cout << "Calculating Vex for orbital eq. " << ko << ", term from k1 = " << k1 << " (full sub-shell)" << std::endl;

      int l1 = _o[k1].initialL();
      int m1 = _o[k1].initialM();


      std::vector<ldouble> vex(_g.N(), 0); // calculate it here first

      ldouble Q = 0;
      std::vector<ldouble> E(_g.N(), 0); // electric field
      for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
        ldouble r2 = _g(ir2);
        ldouble dr = 0;
        if (ir2 < _g.N()-1) dr = _g(ir2+1) - _g(ir2);
        Q += (_o[k1].getNorm(ir2, l1, m1, _g)*_o[ko].getNorm(ir2, lj, mj, _g))*std::pow(r2, 2)*dr;
        E[ir2] = Q/std::pow(r2, 2);
      }
      vex[_g.N()-1] = Q/_g(_g.N()-1);
      for (int ir2 = _g.N()-2; ir2 >= 0; --ir2) {
        ldouble dr = 0;
        if (ir2 < _g.N()-1) dr = _g(ir2+1) - _g(ir2);
        vex[ir2] = vex[ir2+1] + E[ir2]*dr;
      }

      for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
        _vexsum[std::pair<int,int>(ko, k1)][std::pair<int,int>(lj, mj)][ir2] += vex[ir2];
      }
    }
  }

  /*
  for (int ko = 0; ko < _o.size(); ++ko) {
    int lj = _o[ko].initialL();
    int mj = _o[ko].initialM();
    // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
    for (int lj = 0; lj < _o[ko].L()+1; ++lj) { // loop over l in Y_j
      for (int mj = -lj; mj < lj+1; ++mj) { // loop over m in Y_j
        for (auto k1 : not_done) {
          std::cout << "Calculating Vex for orbital eq. " << ko << ", term from k1 = " << k1 << " (partial sub-shell)" << std::endl;

          std::vector<ldouble> &vexsum_curr = _vexsum[std::pair<int,int>(ko,k1)][std::pair<int,int>(lj, mj)];

          // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
          for (int l1 = 0; l1 < _o[k1].L()+1; ++l1) { // loop over l in Y_j
            for (int m1 = -l1; m1 < l1+1; ++m1) { // loop over m in Y_j

              if (_o[k1].spin()*_o[ko].spin() < 0) continue;

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
                    T2 = std::sqrt((2*l+1)/(4*M_PI))*CG(l, lj, 0, 0, lj, 0)*CG(l, lj, m, -mj, lj, -mj);
                    //if (m == 0 && l == 0) T2 = std::pow(4*M_PI, -0.5);
                    T += T1*T2;
                  }
                  vexsum_curr[ir2] += beta*T;
                } // for each r in Vex integration
              } // for each l in the 1/|r1 - r2| expansion in sph. harm.

            } // for each l1 of the orbital basis expansion
          } // for each orbital in the Vex sum

        } // for each Vd term in a unique equation coming from the multiplication by Y*_i
      } // for each m in the orbital basis expansion
    } // for each l in the orbital basis expansion

  }
  */

  for (int ko = 0; ko < _o.size(); ++ko) {
    for (int lj = 0; lj < _o[ko].L()+1; ++lj) { // loop over l in Y_j
      for (int mj = -lj; mj < lj+1; ++mj) { // loop over m in Y_j
        for (auto k1 : done) {
          std::vector<ldouble> &currentVex = _vex[std::pair<int,int>(ko,k1)][std::pair<int,int>(lj, mj)];
          for (int k = 0; k < _g.N(); ++k) currentVex[k] = (1-gamma)*currentVex[k] + gamma*_vexsum[std::pair<int,int>(ko,k1)][std::pair<int,int>(lj,mj)][k];
        }
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
    _vdsum[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g.N(), 0);
    _vdsum[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g.N(), 0);
    _vdsum[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g.N(), 0);
    _vdsum[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g.N(), 0);
  }

  std::vector<int> done;
  std::vector<int> not_done;

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int nSameShell = 0;
    for (int kx = 0; kx < _o.size(); ++kx) {
      if (_o[k1].initialN() == _o[kx].initialN() && _o[k1].initialL() == _o[kx].initialL())
        nSameShell++;
    }
    if (nSameShell == 2*(2*_o[k1].initialL() + 1)) {
      done.push_back(k1);
    } else {
      done.push_back(k1);
      //not_done.push_back(k1);
    }
  }

  std::map<int, std::map<std::pair<int,int>, std::vector<ldouble> > > vdsum; // calculate it here first

  // calculate it first with filled orbitals
  // loop over orbitals (this is the sum over k1 above)
  for (auto k1 : done) {
    int l1 = _o[k1].initialL();
    int m1 = _o[k1].initialM();
    std::cout << "Calculating Vd term from k1 = " << k1 << " (filled sub-shell)" << std::endl;

    // temporary variable
    std::vector<ldouble> vd(_g.N(), 0); // calculate it here first

    ldouble Q = 0;
    std::vector<ldouble> E(_g.N(), 0); // electric field
    for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
      ldouble r2 = _g(ir2);
      ldouble dr = 0;
      if (ir2 < _g.N()-1) dr = _g(ir2+1) - _g(ir2);
      Q += std::pow(_o[k1].getNorm(ir2, l1, m1, _g), 2)*std::pow(r2, 2)*dr;
      E[ir2] = Q/std::pow(r2, 2);
    }
    vd[_g.N()-1] = Q/_g(_g.N()-1);
    for (int ir2 = _g.N()-2; ir2 >= 0; --ir2) {
      ldouble dr = _g(ir2+1) - _g(ir2);
      vd[ir2] = vd[ir2+1] + E[ir2]*dr;
    }

    for (int ko = 0; ko < _o.size(); ++ko) {
      if (ko == k1) continue;
      int lj = _o[ko].initialL();
      int mj = _o[ko].initialM();
      for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
        _vdsum[ko][std::pair<int,int>(lj,mj)][ir2] += vd[ir2];
      }
    }
  }

  /*
  // we are looking to calculate Vd(orbital = o, l = vdl, m = vdm)
  // it shows up in this term of the equation:
  // {int [sum_k1 int rpsi_k1(r1)*rpsi_k1(r1) Y_k1(O)*Y_k1(O)/|r1 - r2| dr1 dO] Y*_i(Oo) Y_j(Oo) dOo} rpsi_j(r2)
  // the term Y_j rpsi_j is part of the representation orb_ko = sum_j Y_j rpsi_j(r) of orbital ko
  // Y*_i is the sph. harm. multiplied to the equation and then integrated over to single out one of the sph. harm. terms
  for (int ko = 0; ko < _o.size(); ++ko) {
    // each sub-orbital has a different Ylomo dependency, so there is a different Vd in each case
    for (int lj = 0; lj < _o[ko].L()+1; ++lj) { // loop over l in Y_j
      for (int mj = -lj; mj < lj+1; ++mj) { // loop over m in Y_j

        // now with open shells
        for (auto k1 : not_done) {
          std::cout << "Calculating Vd term for eq. " << ko << ", due to k1 = " << k1 << " (partial sub-shell)" << std::endl;
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
                    T2 = std::sqrt((2*l+1)/(4*M_PI))*CG(l, lj, 0, 0, lj, 0)*CG(l, lj, m, -mj, lj, -mj);
                    //if (m == 0 && l == 0) T2 = std::pow(4*M_PI, -0.5);
                    T += T1*T2;
                  }
                  _vdsum[ko][std::pair<int,int>(lj,mj)][ir2] += beta*T;
                } // for each r in Vd integration
              } // for each l in the 1/|r1 - r2| expansion in sph. harm.
            } // for each m1 of the orbital basis expansion
          } // for each l1 of the orbital basis expansion
        } // for each orbital in the Vd sum
      } // for each m in the orbital basis expansion
    } // for each l in the orbital basis expansion

  }
  */

  for (int ko = 0; ko < _o.size(); ++ko) {
    int lj = _o[ko].initialL();
    int mj = _o[ko].initialM();
    std::cout << "Adding Vd term for eq. " << ko << ", (filled sub-shell)" << std::endl;
    std::vector<ldouble> &currentVd = _vd[ko][std::pair<int,int>(lj, mj)];
    for (int k = 0; k < _g.N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*_vdsum[ko][std::pair<int,int>(lj,mj)][k];
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

ldouble HF::solveForFixedPotentials(int Niter, ldouble F0stop) {
  ldouble gamma = 1; // move in the direction of the negative slope with this velocity per step

  std::string strMethod = "";
  if (_method == 0) {
    strMethod = "Sparse Matrix Numerov";
  } else if (_method == 1) {
    strMethod = "Iterative Numerov with Gordon method for initial condition (http://aip.scitation.org/doi/pdf/10.1063/1.436421)";
  } else if (_method == 2) {
    strMethod = "Iterative Renormalised Numerov (http://aip.scitation.org/doi/pdf/10.1063/1.436421)";
  }

  if (_method == 2) {
    _irs.setFirst();
  }

  ldouble F = 0;
  int nStep = 0;
  while (nStep < Niter) {
    gamma = 0.5*(1 - std::exp(-(nStep+1)/20.0));
    // compute sum of squares of F(x_old)
    nStep += 1;
    if (_method == 0) {
      F = stepSparse(gamma);
    } else if (_method == 1) {
      F = stepGordon(gamma);
    } else if (_method == 2) {
      F = stepRenormalised(gamma);
    }

    // change orbital energies
    std::cout << "Orbital energies at step " << nStep << ", with constraint = " << std::setw(16) << F << ", method = " << strMethod << "." << std::endl;
    std::cout << std::setw(5) << "Index" << " " << std::setw(16) << "Energy (H)" << " " << std::setw(16) << "next energy (H)" << " " << std::setw(16) << "Min. (H)" << " " << std::setw(16) << "Max. (H)" << " " << std::setw(5) << "nodes" << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      ldouble stepdE = _dE[k];
      ldouble newE = (_o[k].E()+stepdE);
      //if (newE > _Emax[k]) newE = 0.5*(_Emax[k] + _Emin[k]);
      //if (newE < _Emin[k]) newE = 0.5*(_Emax[k] + _Emin[k]);
      std::cout << std::setw(5) << k << " " << std::setw(16) << std::setprecision(12) << _o[k].E() << " " << std::setw(16) << std::setprecision(12) << newE << " " << std::setw(16) << std::setprecision(12) << _Emin[k] << " " << std::setw(16) << std::setprecision(12) << _Emax[k] << " " << std::setw(5) << _nodes[k] << std::endl;
      _o[k].E(newE);
    }

    if (std::fabs(*std::max_element(_dE.begin(), _dE.end(), [](ldouble a, ldouble b) -> bool { return std::fabs(a) < std::fabs(b); } )) < F0stop) break;
    //if (std::fabs(F) < F0stop) break;
  }
  return F;
}

void HF::calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E) {
  std::vector<MatrixXld> Lambda(_g.N());
  int N = _om.N();
  F.resize(_g.N());
  K.resize(_g.N());

  for (int i = 0; i < _g.N(); ++i) {
    ldouble r = _g(i);
    F[i].resize(N, N);
    F[i].setZero();
    Lambda[i].resize(N, N);
    Lambda[i].setZero();
    K[i].resize(N, N);
    K[i].setZero();

    for (int idx1 = 0; idx1 < N; ++idx1) {
      int k1 = _om.orbital(idx1);
      int l1 = _om.l(idx1);
      int m1 = _om.m(idx1);

      for (int idx2 = 0; idx2 < N; ++idx2) {
        int k2 = _om.orbital(idx2);
        int l2 = _om.l(idx2);
        int m2 = _om.m(idx2);

        if (idx1 == idx2) {
          ldouble a = 0;
          if (_g.isLog()) a = 2*std::pow(r, 2)*(E[k1] - _pot[i] - _vd[k1][std::pair<int, int>(l1, m1)][i]) - std::pow(l1 + 0.5, 2);
          else a = 2*(E[k1] - _pot[i] - _vd[k1][std::pair<int, int>(l1, m1)][i] - l1*(l1 + 1)/std::pow(_g(i), 2));

          F[i](idx1,idx1) += 1 + a*std::pow(_g.dx(), 2)/12.0;
          Lambda[i](idx1,idx1) += 1 + a*std::pow(_g.dx(), 2)/12.0;
        }
        ldouble vex = _vex[std::pair<int,int>(k1, k2)][std::pair<int, int>(l2, m2)][i];
        ldouble a = 0;

        if (_g.isLog()) a = 2*std::pow(r, 2)*vex;
        else a = 2*vex;

        F[i](idx1,idx2) += a*std::pow(_g.dx(), 2)/12.0;
        if (idx1 != idx2) K[i](idx1, idx2) += a;
        else Lambda[i](idx1, idx2) += a*std::pow(_g.dx(), 2)/12.0;
      }
    }
    //K[i] = F[i].inverse();
    for (int idxD = 0; idxD < N; ++idxD) Lambda[i](idxD, idxD) = 1.0/Lambda[i](idxD, idxD);
    K[i] = Lambda[i]*K[i];
    K[i] = (MatrixXld::Identity(N,N) + std::pow(_g.dx(), 2)/12.0*K[i] + std::pow(_g.dx(), 4)/144.0*(K[i]*K[i]))*Lambda[i];
  }
}

// solve for a fixed energy and calculate _dE for the next step
ldouble HF::stepGordon(ldouble gamma) {
  int N = 0;
  for (int k = 0; k < _o.size(); ++k) {
    N += 2*_o[k].L()+1;
  }

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = -1e-3;
    E[k] = _o[k].E();
    l[k] = _o[k].initialL();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, E);

  ldouble Fn = _igs.solve(E, l, Fmn, Kmn, matched);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k].initialL();
    int m = _o[k].initialM();
    int idx = _om.index(k, l, m);
    for (int i = 0; i < _g.N(); ++i) {
      _o[k](i, l, m) = matched[i](idx);
      if (i >= 10 && _g(i) < std::pow(_o.size(),2) && i < _g.N() - 4 && matched[i](idx)*matched[i-1](idx) <= 0) {
        _nodes[k] += 1;
      }
    }
  }

  VectorXld J(_o.size());
  J.setZero();
  std::cout << "Calculating energy change Jacobian." << std::endl;
  for (int k2 = 0; k2 < _o.size(); ++k2) {
    std::vector<ldouble> EdE = E;
    EdE[k2] += dE[k2];

    std::vector<MatrixXld> Fmd;
    std::vector<MatrixXld> Kmd;
    calculateFMatrix(Fmd, Kmd, EdE);

    ldouble Fd = _igs.solve(EdE, l, Fmd, Kmd, matched);
    J(k2) = (Fd - Fn)/dE[k2];

  }

  ldouble F = Fn;
  for (int k = 0; k < _o.size(); ++k) {
    if (J(k) != 0) {
      _dE[k] = -gamma*Fn/J(k);
    } else {
      _dE[k] = dE[k];
    }
    std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble HF::stepRenormalised(ldouble gamma) {
  int N = _om.N();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = 1e-3;
    E[k] = _o[k].E();
    l[k] = _o[k].initialL();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, E);

  ldouble Fn = _irs.solve(E, l, Fmn, Kmn, matched);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k].initialL();
    int m = _o[k].initialM();
    int idx = _om.index(k, l, m);
    for (int i = 0; i < _g.N(); ++i) {
      _o[k](i, l, m) = matched[i](idx);
      if (i >= 10 && _g(i) < std::pow(_o.size(),2) && i < _g.N() - 4 && matched[i](idx)*matched[i-1](idx) <= 0) {
        _nodes[k] += 1;
      }
    }
  }

  VectorXld grad(_o.size());
  grad.setZero();
  for (int k = 0; k < _o.size(); ++k) {
    std::vector<ldouble> EdE = E;
    EdE[k] += dE[k];

    std::vector<MatrixXld> Fmd;
    std::vector<MatrixXld> Kmd;
    calculateFMatrix(Fmd, Kmd, EdE);

    ldouble Fd = _irs.solve(EdE, l, Fmd, Kmd, matched);
    grad(k) = (Fd - Fn)/dE[k];
  }

  /*
  // calculate Hessian(F) * dX = - grad(F)
  // this does not work if looking for an asymptotic minimum!
  // H(k1, k2) = [ (F(E+dE2+dE1) - F(E+dE2))/dE1 - (F(E+dE1) - F(E))/dE1 ]/dE2
  MatrixXld H(_o.size(), _o.size());
  H.setZero();
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    ldouble der1 = grad(k1); // derivative at F(E) w.r.t. E1
    // move E(k2) by delta E(k2) and calculate derivative there
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      std::vector<ldouble> EdE = E;
      EdE[k2] += dE[k2]; // move to this other point

      std::vector<MatrixXld> Fmd;
      std::vector<MatrixXld> Kmd;
      calculateFMatrix(Fmd, Kmd, EdE);
      ldouble Fd21 = _irs.solve(EdE, l, Fmd, Kmd, matched); // nominal at E(k2)+dE(k2)

      EdE[k1] += dE[k1]; // variation due to E(k1)

      calculateFMatrix(Fmd, Kmd, EdE);
      ldouble Fd22 = _irs.solve(EdE, l, Fmd, Kmd, matched);

      ldouble der2 = (Fd22 - Fd21)/(dE[k1]); // derivative at E(k2)+dE(k2)

      H(k1, k2) = (der2 - der1)/dE[k2];
    }
  }
  std::cout << "Hessian: " << std::endl << H << std::endl;
  std::cout << "grad: " << std::endl << grad << std::endl;
  VectorXld dX(_o.size());
  dX.setZero();
  JacobiSVD<MatrixXld> decH(H, ComputeThinU | ComputeThinV);
  if (H.determinant() != 0) {
    dX = decH.solve(-grad);
  }
  */
  ldouble F = Fn;
  for (int k = 0; k < _o.size(); ++k) {
    //_dE[k] = gamma*dX(k); // to use the curvature for extrema finding
    _dE[k] = -gamma*Fn/grad(k); // for root finding
    if (std::fabs(_dE[k]) > 0.1) _dE[k] = 0.1*_dE[k]/std::fabs(_dE[k]);
    std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble HF::stepSparse(ldouble gamma) {
  // 1) build sparse matrix _A
  // 2) build sparse matrix _b
  _lsb.prepareMatrices(_A, _b0, _o, _pot, _vd, _vex, _g);
  //std::cout << _A << std::endl;
  //std::cout << _b0 << std::endl;
  // 3) solve sparse system
  _b.resize(_b0.rows(), 1);
  ConjugateGradient<SMatrixXld, Upper> solver;
  //SparseQR<SMatrixXld, COLAMDOrdering<int> > solver;
  solver.compute(_A);
  _b = solver.solve(_b0);

  //SMatrixXld L(_b.rows(), _b.rows());
  //for (int idxD = 0; idxD < _b.rows(); ++idxD) L.coeffRef(idxD, idxD) = _A.coeffRef(idxD, idxD);
  //SMatrixXld K = _A - L;
  //for (int idxD = 0; idxD < _b.rows(); ++idxD) L.coeffRef(idxD, idxD) = 1.0/L.coeffRef(idxD, idxD);
  //K = L*K;
  //SMatrixXld I(_b.rows(), _b.rows());
  //I.setIdentity();
  //K = (I + K + (K*K))*L;
  //_b = K*_b0;

  //std::cout << "b:" << _b << std::endl;
  
  // 4) change results in _o[k]
  _lsb.propagate(_b, _o, _dE, _g, gamma);
  // 5) change results in _dE[k]

  // count nodes for monitoring
  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k].initialL();
    int m = _o[k].initialM();
    for (int i = 0; i < _g.N(); ++i) {
      if (i >= 10 && _g(i) < std::pow(_o.size(),2) && i < _g.N() - 4 && _o[k](i, l, m)*_o[k](i-1, l, m) <= 0) {
        _nodes[k] += 1;
      }
    }
    //if (_nodes[k] < _o[k].initialN() - _o[k].initialL() - 1) {
    //  _dE[k] = std::fabs(_Z*_Z*0.5/std::pow(_o[k].initialN(), 2) - _Z*_Z*0.5/std::pow(_o[k].initialN()+1, 2));
    //} else if (_nodes[k] > _o[k].initialN() - _o[k].initialL() - 1) {
    //  _dE[k] = -std::fabs(_Z*_Z*0.5/std::pow(_o[k].initialN(), 2) - _Z*_Z*0.5/std::pow(_o[k].initialN()+1, 2));
    //}
  }

  // 6) calculate F = sum _b[k]^2
  ldouble F = 0;
  for (int k = 0; k < _b.rows(); ++k) F += std::pow(_b(k), 2);
  return F;
}

void HF::addOrbital(int L, int s, int initial_n, int initial_l, int initial_m) {
  _o.push_back(Orbital(_g.N(), s, L, initial_n, initial_l, initial_m));
  // initialise energies and first solution guess
  for (int k = 0; k < _o.size(); ++k) {
    _o[k].E(-_Z*_Z*0.5/std::pow(_o[k].initialN(), 2));

    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int ir = 0; ir < _g.N(); ++ir) { // for each radial point
          _o[k](ir, l, m) = std::pow(_Z*_g(ir)/((ldouble) _o[k].initialN()), l+0.5)*std::exp(-_Z*_g(ir)/((ldouble) _o[k].initialN()));
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

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vd();
    _vdsum[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g.N(), 0);
    _vdsum[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g.N(), 0);
    _vdsum[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g.N(), 0);
    _vdsum[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g.N(), 0);
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vexsum[std::pair<int, int>(k, k2)] = Vex();
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g.N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g.N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g.N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g.N(), 0);
    }
  }
}



