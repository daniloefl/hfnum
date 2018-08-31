#include "HFS.h"
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

HFS::HFS(ldouble Z)
  : SCF(Z) {
}

HFS::HFS(const std::string fname)
  : SCF() {
  load(fname);
}

HFS::~HFS() {
}

std::vector<ldouble> HFS::getDirectPotential(int k) {
  return _vd[k];
}

std::vector<ldouble> HFS::getExchangePotential(int k, int k2) {
  return _vex[std::pair<int,int>(k,k2)];
}


python::list HFS::getDirectPotentialPython(int k) {
  python::list l;
  for (int i = 0; i < _g->N(); ++i) l.append(_vd[k][i]);
  return l;
}

python::list HFS::getExchangePotentialPython(int k, int k2) {
  python::list l;
  for (int i = 0; i < _g->N(); ++i) l.append(_vex[std::pair<int,int>(k, k2)][i]);
  return l;
}


void HFS::save(const std::string fout) {
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
    f << " " << std::setw(5) << "term" << " " << std::setw(5) << _o[i]->term();
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
  for (auto &i : _lambdaMap) {
    int k1 = i.first % 100;
    int k2 = i.first / 100;
    f << std::setw(10) << "lambdaMap" << " " << std::setw(10) << k1 << " " << std::setw(10) << k2;
    f << " " << std::setw(5) << "value" << std::setw(10) << i.second;
    f << std::endl;
  }
  for (int k = 0; k < _lambda.size(); ++k) {
    f << std::setw(10) << "lambda" << " " << std::setw(10) << k;
    f << " " << std::setw(5) << "value" << std::setw(10) << _lambda[k];
    f << std::endl;
  }
}

void HFS::load(const std::string fin) {
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
  _lambdaMap = sr._lambdaMap;
  _lambda = sr._lambda;
  std::cout << "Vex load" << std::endl;
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
}



ldouble HFS::getE0() {
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    ldouble A = _o[k]->g();
    E0 += A*_o[k]->E();
  }
  ldouble J = 0;
  ldouble K = 0;
  for (auto &vditm : _vd) {
    int k = vditm.first;
    ldouble A = _o[k]->g();
    for (int ir = 0; ir < _g->N()-1; ++ir) {
      ldouble r = (*_g)(ir);
      ldouble rp1 = (*_g)(ir+1);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      ldouble fnp1 = A*_vd[k][ir+1]*std::pow(_o[k]->getNorm(ir+1, *_g), 2)*std::pow(rp1, 2);
      ldouble fn = A*_vd[k][ir]*std::pow(_o[k]->getNorm(ir, *_g), 2)*std::pow(r, 2);
      J += 0.5*(fn+fnp1)*dr;
    }
  }
  for (auto &vexitm : _vex) {
    const int k1 = vexitm.first.first;
    const int k2 = vexitm.first.second;
    ldouble A = _o[k2]->g();
    for (int ir = 0; ir < _g->N()-1; ++ir) {
      ldouble r = (*_g)(ir);
      ldouble rp1 = (*_g)(ir+1);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      ldouble fnp1 = A*_vex[std::pair<int,int>(k1, k2)][ir+1]*_o[k1]->getNorm(ir+1, *_g)*_o[k2]->getNorm(ir+1, *_g)*std::pow(rp1, 2);
      ldouble fn = A*_vex[std::pair<int,int>(k1, k2)][ir]*_o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2);
      K += 0.5*(fn+fnp1)*dr;
    }
  }
  E0 += -0.5*(J - K);
  return E0;
}

void HFS::solve(int NiterSCF, int Niter, ldouble F0stop) {
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
      ldouble a_max = 0;
      for (int i = _g->N()-1; i >= 0; --i) {
        ldouble r = (*_g)(i);
        ldouble a = 0;
        if (_g->isLog()) a = 2*std::pow(r, 2)*(_o[k]->E() - _pot[i] - _vd[k][i] + _vex[std::pair<int,int>(k,k)][i]) - std::pow(lmain_eq + 0.5, 2);
        else a = 2*(_o[k]->E() - _pot[i] - _vd[k][i] + _vex[std::pair<int,int>(k,k)][i]) - lmain_eq*(lmain_eq+1)/std::pow(r, 2);
        if (std::fabs(a) > a_max) a_max = std::fabs(a);
      }
      ldouble a_m1 = 0;
      //for (int i = 3; i < _g->N()-3; ++i) {
      for (int i = _g->N()-3; i >= 3; --i) {
        ldouble r = (*_g)(i);
        ldouble a = 0;
        if (_g->isLog()) a = 2*std::pow(r, 2)*(_o[k]->E() - _pot[i] - _vd[k][i]) - std::pow(lmain_eq + 0.5, 2);
        else a = 2*(_o[k]->E() - _pot[i] - _vd[k][i]) - lmain_eq*(lmain_eq+1)/std::pow(r, 2);

        if (std::fabs(a) > 0.05*a_max) {
          if (icl[k] < 0 && a*a_m1 < 0) {
            icl[k] = i;
            break;
          }
          a_m1 = a;
        }
      }
      if (icl[k] < 0) icl[k] = 10;
      std::cout << "Found classical crossing for orbital " << k << " at " << icl[k] << std::endl;
    }

    for (int k = 0; k < _o.size(); ++k) {
      _nodes[k] = 0;
      _Emin[k] = -_Z*_Z;
      _Emax[k] = 0;
    }

    std::cout << "SCF step " << nStepSCF << std::endl;
    solveForFixedPotentials(Niter, F0stop);
    nStepSCF++;
  }
}

void HFS::calculateY() {
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
        //_Zt[10000*k + 100*k1 + 1*k2][0] = 0;
        //for (int ir = 0; ir < _g->N()-1; ++ir) {
        //  ldouble r = (*_g)(ir);
        //  ldouble rp1 = (*_g)(ir+1);
        //  ldouble x = std::log(r);
        //  ldouble dr = (*_g)(ir+1) - (*_g)(ir);
        //  ldouble dx = std::log((*_g)(ir+1)) - std::log((*_g)(ir));
        //  ldouble fn = std::pow(r, 3)*_o[k1]->getNorm(ir, *_g) * _o[k2]->getNorm(ir, *_g);
        //  ldouble fnp1 = std::pow(rp1, 3)*_o[k1]->getNorm(ir+1, *_g) * _o[k2]->getNorm(ir+1, *_g);
        //  _Zt[10000*k + 100*k1 + 1*k2][ir+1] = std::exp(-dx*k)*_Zt[10000*k + 100*k1 + 1*k2][ir] + 0.5*(fnp1+fn)*std::exp(dx*k)*dx;
        //}
        //_Y[10000*k + 100*k1 + 1*k2][_g->N()-1] = _Zt[10000*k + 100*k1 + 1*k2][_g->N()-1];
        //for (int ir = _g->N()-1; ir >= 1; --ir) {
        //  ldouble r = (*_g)(ir);
        //  ldouble x = std::log(r);
        //  ldouble dr = (*_g)(ir) - (*_g)(ir-1);
        //  ldouble dx = std::log((*_g)(ir)) - std::log((*_g)(ir-1));
        //  ldouble fn = (2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir];
        //  ldouble fnm1 = (2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir-1];
        //  _Y[10000*k + 100*k1 + 1*k2][ir-1] = std::exp(-dx*(k+1))*_Y[10000*k + 100*k1 + 1*k2][ir] + 0.5*(fn+fnm1)*std::exp(-(k+1)*dx)*dx;
        //}
        //for (int ir = 0; ir < _g->N()-1; ++ir) {
        //  ldouble r = (*_g)(ir);
        //  _Y[10000*k + 100*k1 + 1*k2][ir] = _Y[10000*k + 100*k1 + 1*k2][ir]/r;
        //}

        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          // integrate r1 from 0 to r
          for (int ir1 = 0; ir1 < ir; ++ir1) {
            ldouble r1 = (*_g)(ir1);
            ldouble r1p1 = (*_g)(ir1+1);
            ldouble dr1 = (*_g)(ir1+1) - (*_g)(ir1);
            ldouble fn = _o[k1]->getNorm(ir1, *_g) * _o[k2]->getNorm(ir1, *_g) * std::pow(r1/r, k)/r * r1 * r1;
            ldouble fnp1 = _o[k1]->getNorm(ir1+1, *_g) * _o[k2]->getNorm(ir1+1, *_g) * std::pow(r1p1/r, k)/r * r1p1 * r1p1;
            _Y[10000*k + 100*k1 + 1*k2][ir] += 0.5*(fn+fnp1) * dr1;
          }
          // integrate r1 from r to inf
          for (int ir1 = ir; ir1 < _g->N()-1; ++ir1) {
            ldouble r1 = (*_g)(ir1);
            ldouble r1p1 = (*_g)(ir1+1);
            ldouble dr1 = (*_g)(ir1+1) - (*_g)(ir1);
            ldouble fn = _o[k1]->getNorm(ir1, *_g) * _o[k2]->getNorm(ir1, *_g) * std::pow(r/r1, k)/r1 * r1 * r1;
            ldouble fnp1 = _o[k1]->getNorm(ir1+1, *_g) * _o[k2]->getNorm(ir1+1, *_g) * std::pow(r/r1p1, k)/r1p1 * r1p1 * r1p1;
            _Y[10000*k + 100*k1 + 1*k2][ir] += 0.5*(fn+fnp1) * dr1;
          }
        }

      }
    }
  }
}

void HFS::calculateVex(ldouble gamma) {
  std::cout << "Calculating Vex." << std::endl;

  for (int k = 0; k < _o.size(); ++k) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vexsum[std::pair<int, int>(k, k2)] = Vradial(_g->N(), 0);
    }
  }

}

void HFS::calculateVd(ldouble gamma) {
  std::cout << "Calculating Vd." << std::endl;

  Vradial vex(_g->N(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vradial(_g->N(), 0);
  }

  // calculate it first with filled orbitals, dividing by the number of orbitals
  // this is exact if all 2(2*l+1) orbitals in this level are filled
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();
    std::cout << "Calculating Vd term from k = " << k1 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

    std::cout << "Calculating Vd term from k = " << k1 << std::endl;

    for (int k2 = 0; k2 < _o.size(); ++k2) {
      int l2 = _o[k2]->l();
      int m2 = _o[k2]->m();

      // This is the central part
      ldouble A = _o[k2]->g();
      for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
        _vdsum[k1][ir1] += A*_Y[10000*0 + 100*k2 + 1*k2][ir1];
      }

      // from C. Fischer, "The Hartree-Fock method for atoms"
      // Re-estimated in calculations/Angular coefficients Hartree-Fock numerical.ipynb
      // Values agree, but taken in abs value ... how to average them in km?
      // https://journals.aps.org/pr/pdf/10.1103/PhysRev.34.1293
      for (int k = 2; k <= 2*l2; k += 2) {
        ldouble A = 0.0;
        if (k == 2 && l2 == 1 && l1 == 1) {
          for (int ml1_idx = 0; ml1_idx < _o[k1]->term().size(); ++ml1_idx) {
            int ml1 = ml1_idx/2 - 1;
            if (_o[k1]->term()[ml1_idx] != '+' && _o[k1]->term()[ml1_idx] != '-') continue;
            for (int ml2_idx = 0; ml2_idx < _o[k2]->term().size(); ++ml2_idx) {
              int ml2 = ml2_idx/2 - 1;
              if (_o[k2]->term()[ml2_idx] != '+' && _o[k2]->term()[ml2_idx] != '-') continue;
              if (ml1 == -1 && ml2 == -1) A += 1.0/25.0;
              if (ml1 == -1 && ml2 == 0) A += -2.0/25.0;
              if (ml1 == -1 && ml2 == 1) A += 1.0/25.0;
              if (ml1 == 0 && ml2 == -1) A += -2.0/25.0;
              if (ml1 == 0 && ml2 == 0) A += 4.0/25.0;
              if (ml1 == 0 && ml2 == 1) A += -2.0/25.0;
              if (ml1 == 1 && ml2 == -1) A += 1.0/25.0;
              if (ml1 == 1 && ml2 == 0) A += -2.0/25.0;
              if (ml1 == 1 && ml2 == 1) A += 1.0/25.0;
            }
          }
        }
 
        if (A == 0) continue;
        // This is the extra k parts
        for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
          _vdsum[k1][ir1] += A * _Y[10000*k + 100*k2 + 1*k2][ir1];
        }
      }
    }

  }

  for (int ko = 0; ko < _o.size(); ++ko) {
    int lo = _o[ko]->l();
    int mo = _o[ko]->m();
    ldouble A = 0.0;
    ldouble B = 0.0;
    for (int ml_idx = 0; ml_idx < _o[ko]->term().size(); ++ml_idx) {
      if (_o[ko]->term()[ml_idx] == '+') A += 1.0;
      if (_o[ko]->term()[ml_idx] == '-') B += 1.0;
    }
    for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
      vex[ir2] += (A+B)*std::pow(_o[ko]->getNorm(ir2, *_g), 2.0);
    }
  }
  for (int ko = 0; ko < _o.size(); ++ko) {
    for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
      _vdsum[ko][ir2] += -3*3/(8*3.14159)*std::pow(vex[ir2], 1.0/3.0);
    }
  }

  for (int ko = 0; ko < _o.size(); ++ko) {
    std::cout << "Adding Vd term for eq. " << ko << std::endl;
    std::vector<ldouble> &currentVd = _vd[ko];
    for (int k = 0; k < _g->N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*_vdsum[ko][k];
  }
}

void HFS::addOrbital(Orbital *o) {
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




