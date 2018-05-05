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

#include "StateReader.h"

HF::HF()
  : SCF() {
}

HF::HF(const std::string fname)
  : SCF() {
  load(fname);
}

HF::~HF() {
}

std::vector<ldouble> HF::getDirectPotential(int k) {
  return _vd[k];
}

std::vector<ldouble> HF::getExchangePotential(int k, int k2) {
  return _vex[std::pair<int,int>(k,k2)];
}


python::list HF::getDirectPotentialPython(int k) {
  python::list l;
  for (int i = 0; i < _g->N(); ++i) l.append(_vd[k][i]);
  return l;
}

python::list HF::getExchangePotentialPython(int k, int k2) {
  python::list l;
  for (int i = 0; i < _g->N(); ++i) l.append(_vex[std::pair<int,int>(k, k2)][i]);
  return l;
}



void HF::save(const std::string fout) {
  std::ofstream f(fout.c_str());
  f << std::setw(10) << "method" << " " << std::setw(10) << _method << std::endl;
  f << std::setw(10) << "Z" << " " << std::setw(10) << _Z << std::endl;
  f << std::setw(10) << "gamma_scf" << " " << std::setw(10) << _gamma_scf << std::endl;
  f << std::setw(10) << "central" << " " << std::setw(10) << 1 << std::endl;
  f << std::setw(10) << "grid.isLog" << " " << std::setw(10) << _g->type() << std::endl;
  f << std::setw(10) << "grid.dx" << " " << std::setw(10) << _g->dx() << std::endl;
  f << std::setw(10) << "grid.N" << " " << std::setw(10) << _g->N() << std::endl;
  f << std::setw(10) << "grid.rmin" << " " << std::setw(10) << (*_g)(0) << std::endl;
  for (int i = 0; i < _o.size(); ++i) {
    f << std::setw(10) << "orbital" << " " << std::setw(10) << i;
    f << " " << std::setw(5) << "n" << " " << std::setw(5) << _o[i]->n();
    f << " " << std::setw(5) << "l" << " " << std::setw(5) << _o[i]->l();
    f << " " << std::setw(5) << "m" << " " << std::setw(5) << _o[i]->m();
    f << " " << std::setw(5) << "s" << " " << std::setw(5) << _o[i]->spin();
    f << " " << std::setw(5) << "E" << " " << std::setw(64) << std::setprecision(60) << _o[i]->E();
    f << " " << std::setw(5) << "value";
    for (int ir = 0; ir < _g->N(); ++ir) {
      const ldouble v = ((const Orbital) (*_o[i]))(ir);
      f << " " << std::setw(64) << std::setprecision(60) << v;
    }
    f << std::endl;
  }
  for (auto &i : _vd) {
    const int &k = i.first;
    const Vradial &vradial = i.second;
    f << std::setw(10) << "vd" << " " << std::setw(10) << k;
    f << " " << std::setw(5) << "value";
    for (int ir = 0; ir < vradial.size(); ++ir) {
      f << " " << std::setw(64) << std::setprecision(60) << vradial[ir];
    }
    f << std::endl;
  }
  for (auto &i : _vex) {
    const int &k1 = i.first.first;
    const int &k2 = i.first.second;
    const Vradial &vradial = i.second;
    f << std::setw(10) << "vex" << " " << std::setw(10) << k1 << " " << std::setw(10) << k2;
    f << " " << std::setw(5) << "value";
    for (int ir = 0; ir < vradial.size(); ++ir) {
      f << " " << std::setw(64) << std::setprecision(60) << vradial[ir];
    }
    f << std::endl;
  }
}

void HF::load(const std::string fin) {
  std::cout << "Loading state" << std::endl;
  StateReader sr(fin);
  std::cout << "Loaded state" << std::endl;

  _o.clear();
  for (auto &o : _owned_orb) {
    delete o;
  }
  _owned_orb.clear();

  std::cout << "Cleaned" << std::endl;
  
  _method = sr.getInt("method");
  _Z = sr.getDouble("Z");
  _gamma_scf = sr.getDouble("gamma_scf");
  std::cout << "Param load" << std::endl;
  _g->reset((gridType) sr.getInt("grid.isLog"), sr.getDouble("grid.dx"), sr.getInt("grid.N"), sr.getDouble("grid.rmin"));
  std::cout << "Grid reset" << std::endl;
  for (int k = 0; k < sr._o.size(); ++k) {
    _owned_orb.push_back(new Orbital(*sr.getOrbital(k)));
    _o.push_back(_owned_orb[_owned_orb.size()-1]);
  }
  std::cout << "Orbital load" << std::endl;
  for (auto &k : sr._vd) {
    _vd[k.first] = k.second;
  }
  std::cout << "Vd load" << std::endl;
  for (auto &k : sr._vex) {
    _vex[k.first] = k.second;
  }
  std::cout << "Vex load" << std::endl;
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
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
    int l = _o[k]->l();
    int m = _o[k]->m();
    for (int ir = 0; ir < _g->N(); ++ir) {
      ldouble r = (*_g)(ir);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      J += _vd[k][ir]*std::pow(_o[k]->getNorm(ir, *_g), 2)*std::pow(r, 2)*dr;
    }
  }
  for (auto &vexitm : _vex) {
    const int k1 = vexitm.first.first;
    const int k2 = vexitm.first.second;
    for (int ir = 0; ir < _g->N(); ++ir) {
      ldouble r = (*_g)(ir);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      K += _vex[std::pair<int,int>(k1, k2)][ir]*_o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2)*dr;
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
    if (nStepSCF != 0) {
      calculateY();
      calculateVex(_gamma_scf);
      calculateVd(_gamma_scf);
    }

    for (int k = 0; k < _o.size(); ++k) {
      icl[k] = -1;

      ldouble lmain_eq = _o[k]->l();
      int lmain = _o[k]->l();
      int mmain = _o[k]->m();
      // calculate crossing of potential at zero for lmain,mmain
      ldouble a_m1 = 0;
      //for (int i = 3; i < _g->N()-3; ++i) {
      for (int i = _g->N()-3; i >= 3; --i) {
        ldouble r = (*_g)(i);
        ldouble a = 0;
        if (_g->isLog()) a = 2*std::pow(r, 2)*(_o[k]->E() - _pot[i] - _vd[k][i]) - std::pow(lmain_eq + 0.5, 2);
        else a = 2*(_o[k]->E() - _pot[i] - _vd[k][i]) - lmain_eq*(lmain_eq+1)/std::pow(r, 2);
        //if (_g->isLog()) a = 2*std::pow(r, 2)*(_o[k]->E() - _pot[i]) - std::pow(lmain_eq + 0.5, 2);
        //else a = 2*(_o[k]->E() - _pot[i]) - lmain_eq*(lmain_eq+1)/std::pow(r, 2);
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
      _Emin[k] = -_Z*_Z*0.5/std::pow(_o[k]->n(), 2);
      _Emax[k] = 0;
      //_o[k]->E(-0.5*std::pow(_Z/((ldouble) _o[k]->n()), 2));
    }

    std::cout << "SCF step " << nStepSCF << std::endl;
    solveForFixedPotentials(Niter, F0stop);
    nStepSCF++;
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

  std::vector<bool> filled;
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int c = 0;
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (_o[k2]->l() == _o[k1]->l() && _o[k2]->n() == _o[k1]->n()) {
        c++;
      }
    }
    if (c == 2*(2*_o[k1]->l() + 1)) filled.push_back(true);
    else filled.push_back(false);
    std::cout << "calculateVex: Assuming orbital " << k1 << ", filled: " << filled[k1] << std::endl;
  }

  for (int k = 0; k < _o.size(); ++k) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vexsum[std::pair<int, int>(k, k2)] = Vradial(_g->N(), 0);
    }
  }

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();

    // calculate it first with filled orbitals, dividing by the number of orbitals
    // this is exact if all 2(2*l+1) orbitals in this level are filled
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (k1 == k2) continue; // cancels out with Vd, so remove it here and there
      if (_o[k1]->spin()*_o[k2]->spin() < 0) continue; // only applies if same spin, otherwise it is zero

      int l2 = _o[k2]->l();
      int m2 = _o[k2]->m();

      std::cout << "Calculating Vex term from k1 = " << k1 << ", k2 = " << k2 << std::endl;
      for (int k = abs(l1-l2); k <= l1+l2; k += 1) {
        ldouble B = 0.0;
        if (filled[k2]) {
          // exact for a filled shell
          B = 1.0/((ldouble) (2*k + 1))*std::pow(CG(l1, l2, 0, 0, k, 0), 2);
        } else {
          // from C. Fischer, "The Hartree-Fock method for atoms"
          // Re-estimated in calculations/Angular coefficients Hartree-Fock numerical.ipynb
          // Values agree, except for a factor of 1/2 -- from factor of 1/2 in Vex after double counting electrons in summation?
          if (k == 0 && l1 == 0 && l2 == 0) B = 1.0;
          if (k == 0 && l1 == 1 && l2 == 1) B = 1.0/3.0;

          if (k == 1 && l1 == 0 && l2 == 1) B = 1.0/3.0;
          if (k == 1 && l1 == 1 && l2 == 0) B = 1.0/3.0;

          if (k == 2 && l1 == 1 && l2 == 1) B = 2.0/15.0;

          //if (k == 2 && l1 == 0 && l2 == 2) B = 1.0/10.0*2;
          //if (k == 2 && l1 == 2 && l2 == 0) B = 1.0/10.0*2;

          //if (k == 1 && l1 == 1 && l2 == 2) B = 1.0/15.0*2;
          //if (k == 1 && l1 == 2 && l2 == 1) B = 1.0/15.0*2;
          //if (k == 3 && l1 == 1 && l2 == 2) B = 3.0/70.0*2;
          //if (k == 3 && l1 == 2 && l2 == 1) B = 3.0/70.0*2;

          //if (k == 0 && l1 == 2 && l2 == 2) B = 1.0/10.0*2;
          //if (k == 2 && l1 == 2 && l2 == 2) B = 1.0/35.0*2;
          //if (k == 4 && l1 == 2 && l2 == 2) B = 1.0/35.0*2;
        }

        if (B == 0) continue;
        // This is the extra k parts
        for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
          ldouble r1 = (*_g)(ir1);
          _vexsum[std::pair<int,int>(k1, k2)][ir1] += B * _Y[10000*k + 100*k1 + 1*k2][ir1];
        }
      }

      /*
      // temporary variable
      std::vector<ldouble> vex(_g->N(), 0); // calculate it here first
      for (int L = (int) std::fabs(l1 - l2); L <= l1 + l2; ++L) {
        ldouble coeff = 1.0/((ldouble) (2*L + 1))*std::pow(CG(l1, l2, 0, 0, L, 0), 2);
        for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
          ldouble r1 = (*_g)(ir1);
          for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
            ldouble r2 = (*_g)(ir2);
            ldouble dr = 0;
            if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);

            ldouble rsmall = r1;
            ldouble rlarge = r2;
            if (r2 < r1) {
              rsmall = r2;
              rlarge = r1;
            }

            vex[ir1] += coeff*_o[k1]->getNorm(ir2, *_g)*_o[k2]->getNorm(ir2, *_g)*std::pow(r2, 2)*std::pow(rsmall, L)/std::pow(rlarge, L+1)*dr;
          }
        }
      }

      for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
        _vexsum[std::pair<int,int>(k1, k2)][ir1] += vex[ir1];
      }
      */
    }
  }

  for (int ko = 0; ko < _o.size(); ++ko) {
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      std::vector<ldouble> &currentVex = _vex[std::pair<int,int>(ko,k1)];
      for (int k = 0; k < _g->N(); ++k) currentVex[k] = (1-gamma)*currentVex[k] + gamma*_vexsum[std::pair<int,int>(ko,k1)][k];
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
void HF::calculateY() {
  std::cout << "Calculating Y" << std::endl;
  // Calculating Y_k(orb1, orb2)[r]
  // index in Y is 10000*k + 100*orb1 + orb2
  for (int k = 0; k <= 2; ++k) {
    std::cout << "Calculating Y for k "<< k << std::endl;
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      int l1 = _o[k1]->l();
      int m1 = _o[k1]->m();
      for (int k2 = 0; k2 < _o.size(); ++k2) {
        int l2 = _o[k2]->l();
        int m2 = _o[k2]->m();
        _Y[10000*k + 100*k1 + 1*k2] = Vradial(_g->N(), 0);
        _Zt[10000*k + 100*k1 + 1*k2] = Vradial(_g->N(), 0);

        // r Y(r) = int_0^r Pk1*t Pk2*t (t/r)^k dt +
        //         int_r^inf Pk1*t Pk2*t (r/t)^(k+1) dt
        // Z(r) = int_0^r Pk1*t Pk2*t (t/r)^k dt
        // dZ/dr = Pk1*r Pk2*r - k/r Z
        // d(rY)/dr = 1/r [ (k+1) (rY) - (2k + 1) Z ]
        // Z (r=0) = 0
        // lim Y when r -> infinity = Z
        // define r = exp(x), x = ln(r)
        // dZ/dx = dZ/dr dr/dx = Pk1*r Pk2*r * r - k Z
        // d(rY)/dx = [ (k+1) (rY) - (2k + 1) Z ]
        // d(exp(kx)*Z)/dx = k exp(kx) Z + exp(kx) dZ/dx
        //                 = exp(kx) *Pk1*r Pk2 *r *r
        // d(exp(- (k+1)x) (rY))/dx = -(k+1) exp(-(k+1)x) (rY) + exp(-(k+1)x) d(rY)/dx
        //                       = - (2k +1) Z exp(-(k+1)x)
        // Solution for:
        // dy/dx = f(y, x)
        // y(x) = y(x-1) + dx/12 (5 f(n) + 8 f(n-1) - f(n-2))
        // With y(0) = f(0) = f(-1) = 0
        _Zt[10000*k + 100*k1 + 1*k2][0] = 0;
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble x = std::log(r);
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble dx = std::log((*_g)(ir+1)) - std::log((*_g)(ir));
          _Zt[10000*k + 100*k1 + 1*k2][ir+1] = std::exp(-dx*k)*_Zt[10000*k + 100*k1 + 1*k2][ir] + std::pow(r, 3)*_o[k1]->getNorm(ir, *_g) * _o[k2]->getNorm(ir, *_g)*std::exp(dx*k)*dx;
        }
        /*
        // solving it using a better integration method
        // solve for exp(kx)*Z first
        _Zt[10000*k + 100*k1 + 1*k2][0] = 0;
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble rm1 = (*_g)(ir-1);
          ldouble rp1 = (*_g)(ir+1);
          ldouble x = std::log(r);
          ldouble xm1 = std::log(rm1);
          ldouble xp1 = std::log(rp1);
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble dx = std::log((*_g)(ir+1)) - std::log((*_g)(ir));
          ldouble fn = std::exp(k*x)*std::pow(r, 3)*_o[k1]->getNorm(ir, *_g) * _o[k2]->getNorm(ir, *_g);
          ldouble fnp1 = std::exp(k*xp1)*std::pow(rp1, 3)*_o[k1]->getNorm(ir+1, *_g) * _o[k2]->getNorm(ir+1, *_g);
          ldouble fnm1 = std::exp(k*xm1)*std::pow(rm1, 3)*_o[k1]->getNorm(ir-1, *_g) * _o[k2]->getNorm(ir-1, *_g);
          _Zt[10000*k + 100*k1 + 1*k2][ir+1] = _Zt[10000*k + 100*k1 + 1*k2][ir] + fn*dx;
        }
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          ldouble x = std::log(r);
          _Zt[10000*k + 100*k1 + 1*k2][ir] = std::exp(-x*k)*_Zt[10000*k + 100*k1 + 1*k2][ir];
        }
        */
        _Y[10000*k + 100*k1 + 1*k2][_g->N()-1] = _Zt[10000*k + 100*k1 + 1*k2][_g->N()-1];
        for (int ir = _g->N()-1; ir >= 1; --ir) {
          ldouble r = (*_g)(ir);
          ldouble x = std::log(r);
          ldouble dr = (*_g)(ir) - (*_g)(ir-1);
          ldouble dx = std::log((*_g)(ir)) - std::log((*_g)(ir-1));
          _Y[10000*k + 100*k1 + 1*k2][ir-1] = std::exp(-dx*(k+1))*_Y[10000*k + 100*k1 + 1*k2][ir] + (2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir]*std::exp(-(k+1)*dx)*dx;
        }
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          _Y[10000*k + 100*k1 + 1*k2][ir] = _Y[10000*k + 100*k1 + 1*k2][ir]/r;
        }
        /*
        _Y[10000*k + 100*k1 + 1*k2][_g->N()-1] = _Zt[10000*k + 100*k1 + 1*k2][_g->N()-1];
        for (int ir = _g->N()-1; ir >= 1; --ir) {
          ldouble r = (*_g)(ir);
          ldouble rm1 = (*_g)(ir-1);
          ldouble rp1 = (*_g)(ir+1);
          ldouble x = std::log(r);
          ldouble xm1 = std::log(rm1);
          ldouble xp1 = std::log(rp1);
          ldouble dr = (*_g)(ir) - (*_g)(ir-1);
          ldouble dx = std::log((*_g)(ir)) - std::log((*_g)(ir-1));
          ldouble fnp1 = -(2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir+1]*std::exp(-(k+1)*xp1);
          ldouble fn = -(2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir]*std::exp(-(k+1)*x);
          ldouble fnm1 = -(2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir-1]*std::exp(-(k+1)*xm1);
          _Y[10000*k + 100*k1 + 1*k2][ir-1] = _Y[10000*k + 100*k1 + 1*k2][ir] - fn*dx;
        }
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          ldouble x = std::log(r);
          _Y[10000*k + 100*k1 + 1*k2][ir] = std::exp(x*(k+1))*_Y[10000*k + 100*k1 + 1*k2][ir]/r;
        }
        */


        // classical integration:
        /*
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);

          // integrate r1 from 0 to r
          for (int ir1 = 0; ir1 < ir; ++ir1) {
            ldouble r1 = (*_g)(ir1);
            ldouble dr1 = (*_g)(ir1+1) - (*_g)(ir1);
            _Y[10000*k + 100*k1 + 1*k2][ir] += _o[k1]->getNorm(ir1, *_g) * _o[k2]->getNorm(ir1, *_g) * std::pow(r1/r, k)/r * r1 * r1 * dr1;
          }

          // integrate r1 from r to inf
          for (int ir1 = ir; ir1 < _g->N()-1; ++ir1) {
            ldouble r1 = (*_g)(ir1);
            ldouble dr1 = (*_g)(ir1+1) - (*_g)(ir1);
            _Y[10000*k + 100*k1 + 1*k2][ir] += _o[k1]->getNorm(ir1, *_g) * _o[k2]->getNorm(ir1, *_g) * std::pow(r/r1, k)/r1 * r1 * r1 * dr1;
          }
        }
        */

      }
    }
  }
}

void HF::calculateVd(ldouble gamma) {
  std::cout << "Calculating Vd." << std::endl;

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vradial(_g->N(), 0);
  }

  std::vector<bool> filled;
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int c = 0;
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (_o[k2]->l() == _o[k1]->l() && _o[k2]->n() == _o[k1]->n()) {
        c++;
      }
    }
    if (c == 2*(2*_o[k1]->l() + 1)) filled.push_back(true);
    else filled.push_back(false);
    std::cout << "calculateVd: Assuming orbital " << k1 << ", filled: " << filled[k1] << std::endl;
  }

  // calculate it first with filled orbitals, dividing by the number of orbitals
  // this is exact if all 2(2*l+1) orbitals in this level are filled
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();

    std::cout << "Calculating Vd term from k = " << k1 << std::endl;

    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (k2 == k1) continue; // cancels out with Vex ... we also remove it from there

      int l2 = _o[k2]->l();
      int m2 = _o[k2]->m();

      // This is the central part
      for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
        _vdsum[k1][ir1] += _Y[10000*0 + 100*k2 + 1*k2][ir1];
      }

      if (filled[k2]) continue; // the following is an approximation in case of not-filled shells

      // from C. Fischer, "The Hartree-Fock method for atoms"
      // Re-estimated in calculations/Angular coefficients Hartree-Fock numerical.ipynb
      // Values agree, but taken in abs value ... how to average them in km?
      for (int k = 2; k <= 2*l2; k += 2) {
        ldouble A = 0.0;
        if (k == 2 && l2 == 1) A = 2.0/25.0;

        if (k == 2 && l2 == 2) A = 2.0/63.0;
        if (k == 4 && l2 == 2) A = 2.0/63.0;
        if (k == 2 && l2 == 3) A = 4.0/195.0;
        if (k == 4 && l2 == 3) A = 2.0/143.0;
        if (k == 6 && l2 == 3) A = 100.0/5577.0;
 
        if (A == 0) continue;
        // This is the extra k parts
        for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
          _vdsum[k1][ir1] += A * _Y[10000*k + 100*k1 + 1*k1][ir1];
        }
      } // end of vdsum averaging over angles

      
    }

    /*
    int lmax = 2;
    // temporary variable
    std::vector<ldouble> vd(_g->N(), 0); // calculate it here first

    for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
      ldouble r1 = (*_g)(ir1);
      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        ldouble r2 = (*_g)(ir2);
        ldouble dr = 0;
        if (ir2 < _g->N()-1) dr = (*_g)(ir2+1) - (*_g)(ir2);

        // this assumes filled shells and averages over them
        // works well for s shells, but not p-shells
        //ldouble rmax = r1;
        //if (ir2 > ir1) rmax = r2;
        //vd[ir1] += std::pow(_o[k1]->getNorm(ir2, l1, m1, *_g), 2)*std::pow(r2, 2)/rmax*dr;

        ldouble rsmall = r1;
        ldouble rlarge = r2;
        if (r2 < r1) {
          rsmall = r2;
          rlarge = r1;
        }
        int l = 0;
        int m = 0;
        //for (int l = 0; l <= lmax; ++l) {
        //  for (int m = -l; m <= l; ++m) {
            // first 1/sqrt(PI) is the average of spherical harmonic in T2 integrated in Omega2
            //vd[ir1] += 1.0/std::sqrt(4*M_PI)*(2*l1+1.0)/std::sqrt(4*M_PI*(2*l+1.0))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, m1, m1, l, m)*(4*M_PI/(2*l+1.0))*std::pow(_o[k1]->getNorm(ir2, l1, m1, *_g), 2)*std::pow(r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr;
            // simplified:
            vd[ir1] += (2*l1+1.0)/std::sqrt(2*l+1.0)*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, m1, m1, l, m)*(1.0/(2*l+1.0))*std::pow(_o[k1]->getNorm(ir2, *_g), 2)*std::pow(r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr;
        //  }
        //}
        //int l = 0;
        //int m = 0;
        //vd[ir1] += (2.0*l1+1.0)*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, m1, m1, l, m)*std::pow(_o[k1]->getNorm(ir2, l1, m1, *_g), 2)*std::pow(r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr;
      }
    }

    for (int ko = 0; ko < _o.size(); ++ko) {
      for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
        _vdsum[ko][ir2] += vd[ir2];
      }
    }
     */
  }

  for (int ko = 0; ko < _o.size(); ++ko) {
    std::cout << "Adding Vd term for eq. " << ko << std::endl;
    std::vector<ldouble> &currentVd = _vd[ko];
    for (int k = 0; k < _g->N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*_vdsum[ko][k];
  }
}


void HF::calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<MatrixXld> &C,std::vector<ldouble> &E) {
  std::vector<MatrixXld> Lambda(_g->N());
  int N = _om.N();
  F.resize(_g->N());
  K.resize(_g->N());
  C.resize(_g->N());

  for (int i = 0; i < _g->N(); ++i) {
    ldouble r = (*_g)(i);
    F[i].resize(N, N);
    F[i].setZero();
    Lambda[i].resize(N, N);
    Lambda[i].setZero();
    K[i].resize(N, N);
    K[i].setZero();
    C[i].resize(N, N);
    C[i].setZero();

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
          if (_g->isLog()) a = 2*std::pow(r, 2)*(E[k1] - _pot[i] - _vd[k1][i] + _vex[std::pair<int,int>(k1, k2)][i]) - std::pow(l1_eq + 0.5, 2);
          else a = 2*(E[k1] - _pot[i] - _vd[k1][i] + _vex[std::pair<int,int>(k1, k2)][i] - l1_eq*(l1_eq + 1)/std::pow((*_g)(i), 2));

          F[i](idx1,idx1) += 1 + a*std::pow(_g->dx(), 2)/12.0;
          Lambda[i](idx1,idx1) += 1 + a*std::pow(_g->dx(), 2)/12.0;
        } else {
          ldouble vex = _vex[std::pair<int,int>(k1, k2)][i]; // *_o[k2]->getNorm(i, l2, m2, *_g);
          ldouble a = 0;

          if (_g->isLog()) a = 2*std::pow(r, 2)*vex; // *std::pow(r, 0.5);
          else a = 2*vex;

          F[i](idx1,idx2) += a*std::pow(_g->dx(), 2)/12.0;
          Lambda[i](idx1,idx2) += 1 + a*std::pow(_g->dx(), 2)/12.0;
        }
      }
    }
    K[i] = F[i].inverse();
    //for (int idxD = 0; idxD < N; ++idxD) Lambda[i](idxD, idxD) = 1.0/Lambda[i](idxD, idxD);
    //K[i] = Lambda[i]*K[i];
    //K[i] = (MatrixXld::Identity(N,N) + std::pow(_g->dx(), 2)/12.0*K[i] + std::pow(_g->dx(), 4)/144.0*(K[i]*K[i]))*Lambda[i];
  }
}

void HF::addOrbital(Orbital *o) {
  o->N(_g->N());
  _o.push_back(o);
  // initialise energies and first solution guess
  for (int k = 0; k < _o.size(); ++k) {
    _o[k]->E(-_Z*_Z*0.5/std::pow(_o[k]->n(), 2));

    for (int ir = 0; ir < _g->N(); ++ir) { // for each radial point
      (*_o[k])(ir) = std::pow(_Z*(*_g)(ir)/((ldouble) _o[k]->n()), _o[k]->l()+0.5)*std::exp(-_Z*(*_g)(ir)/((ldouble) _o[k]->n()));
    }
  }
  _vd.clear();
  _vex.clear();
  for (int k = 0; k < _o.size(); ++k) {
    _vd[k] = Vradial(_g->N(), 0);
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vex[std::pair<int, int>(k, k2)] = Vradial(_g->N(), 0);
    }
  }

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vradial(_g->N(), 0);
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vexsum[std::pair<int, int>(k, k2)] = Vradial(_g->N(), 0);
    }
  }
}




