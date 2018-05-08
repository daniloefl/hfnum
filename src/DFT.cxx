#include "DFT.h"
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

DFT::DFT()
  : SCF() {
  _isSpinDependent = true;
}

DFT::DFT(const std::string fname)
  : SCF() {
  load(fname);
  _isSpinDependent = true;
}

DFT::~DFT() {
}

std::vector<ldouble> DFT::getDensityUp() {
  return _n_up;
}

python::list DFT::getDensityUpPython() {
  python::list l;
  for (int k = 0; k < _g->N(); ++k) l.append(_n_up[k]);
  return l;
}

python::list DFT::getHartreePython() {
  python::list l;
  for (int k = 0; k < _g->N(); ++k) l.append(_u[k]);
  return l;
}

python::list DFT::getExchangeUpPython() {
  python::list l;
  for (int k = 0; k < _g->N(); ++k) l.append(_vex_lda_up[k]);
  return l;
}

python::list DFT::getExchangeDownPython() {
  python::list l;
  for (int k = 0; k < _g->N(); ++k) l.append(_vex_lda_dw[k]);
  return l;
}

std::vector<ldouble> DFT::getDensityDown() {
  return _n_dw;
}

python::list DFT::getDensityDownPython() {
  python::list l;
  for (int k = 0; k < _g->N(); ++k) l.append(_n_dw[k]);
  return l;
}

void DFT::save(const std::string fout) {
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
  f << std::setw(10) << "n_up";
  f << " " << std::setw(5) << "value";
  for (int ir = 0; ir < _n_up.size(); ++ir) {
    f << " " << std::setw(64) << std::setprecision(60) << _n_up[ir];
  }
  f << std::endl;
  f << std::setw(10) << "n_dw";
  f << " " << std::setw(5) << "value";
  for (int ir = 0; ir < _n_dw.size(); ++ir) {
    f << " " << std::setw(64) << std::setprecision(60) << _n_dw[ir];
  }
  f << std::endl;
}

void DFT::load(const std::string fin) {
  std::ifstream f(fin.c_str());
  std::string line;

  int g_isLog = 1;
  ldouble g_dx = 1e-1;
  int g_N = 220;
  ldouble g_rmin = 1e-6;

  std::string trash;

  _o.clear();
  for (auto &o : _owned_orb) {
    delete o;
  }
  _owned_orb.clear();
  
  while(std::getline(f, line)) {
    std::stringstream ss;
    ss.str(line);

    std::string mode;

    ss >> mode;
    if (mode == "method")
      ss >> _method;
    else if (mode == "Z")
      ss >> _Z;
    else if (mode == "gamma_scf")
      ss >> _gamma_scf;
    else if (mode == "central")
      ss >> trash;
    else if (mode == "grid.isLog")
      ss >> g_isLog;
    else if (mode == "grid.dx")
      ss >> g_dx;
    else if (mode == "grid.N")
      ss >> g_N;
    else if (mode == "grid.rmin")
      ss >> g_rmin;
    else if (mode == "orbital") {
      int io;
      ss >> io;

      std::string trash;

      int o_N, o_L, o_M, o_S, o_G;
      ss >> trash >> o_N >> trash >> o_L >> trash >> o_M >> trash >> o_S >> trash >> o_G;

      ldouble o_E;
      ss >> trash >> o_E;


      _owned_orb.push_back(new Orbital(o_S, o_N, o_L, o_M));
      _o.push_back(_owned_orb[_owned_orb.size()-1]);
      int k = _o.size()-1;
      _o[k]->N(g_N);
      _o[k]->E(o_E);

      ss >> trash;
      ldouble read_value;
      for (int ir = 0; ir < g_N; ++ir) { // for each radial point
        ss >> read_value;
        (*_o[k])(ir) = read_value;
      }
    } else if (mode == "n_up") {
      std::string trash;
      int v_l, v_m;

      ss >> trash;

      _n_up = std::vector<ldouble>(g_N, 0);
      ldouble read_value;
      for (int k = 0; k < g_N; ++k) {
        ss >> read_value;
        _n_up[k] = read_value;
      }
    } else if (mode == "n_dw") {
      std::string trash;

      int v_l, v_m;

      ss >> trash;

      _n_dw = std::vector<ldouble>(g_N, 0);
      ldouble read_value;
      for (int k = 0; k < g_N; ++k) {
        ss >> read_value;
        _n_dw[k] = read_value;
      }
    }
  }
  std::cout << "Load resetting grid with isLog = " << g_isLog << ", dx = " << g_dx << ", g_N = " << g_N << ", g_rmin = " << g_rmin << std::endl;
  _g->reset((gridType) g_isLog, g_dx, g_N, g_rmin);
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }

  _u = std::vector<ldouble>(_g->N(), 0);
  _vex_lda_up = std::vector<ldouble>(_g->N(), 0);
  _vex_lda_dw = std::vector<ldouble>(_g->N(), 0);
  _vsum_up = std::vector<ldouble>(_g->N(), 0);
  _vsum_dw = std::vector<ldouble>(_g->N(), 0);

  calculateV(1.0);
}

ldouble DFT::getE0() {
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    E0 += _o[k]->E();
  }
  ldouble J = 0;
  ldouble vxcn = 0;
  ldouble Exc = 0;
  ldouble Ax = -3.0/4.0*std::pow(3.0/M_PI, 1.0/3.0);
  for (int ir = 0; ir < _g->N(); ++ir) {
    ldouble r = (*_g)(ir);
    ldouble dr = 0;
    if (ir < _g->N()-1)
      dr = (*_g)(ir+1) - (*_g)(ir);
    J += _u[ir]*(_n_up[ir] + _n_dw[ir])*std::pow(r, 2)*dr;
    //vxcn += (_vex_lda_up[ir] + _vex_lda_dw[ir])*(_n_up[ir] + _n_dw[ir])*std::pow(r, 2)*dr;
    vxcn += _vex_lda_up[ir]*_n_up[ir]*std::pow(r, 2)*dr;
    vxcn += _vex_lda_dw[ir]*_n_dw[ir]*std::pow(r, 2)*dr;
    //Exc += 0.5*Ax*std::pow(2*_n_up[ir] + 2*_n_dw[ir], 1.0/3.0)*std::pow(r, 2)*dr;
    Exc += Ax*std::pow(_n_up[ir], 4.0/3.0)*std::pow(r, 2)*dr;
    Exc += Ax*std::pow(_n_dw[ir], 4.0/3.0)*std::pow(r, 2)*dr;
  }
  E0 += -0.5*J - vxcn + Exc;
  return E0;
}

void DFT::solve(int NiterSCF, int Niter, ldouble F0stop) {
  _dE.resize(_o.size());
  _nodes.resize(_o.size());
  _Emax.resize(_o.size());
  _Emin.resize(_o.size());
  icl.resize(_o.size());

  int nStepSCF = 0;
  while (nStepSCF < NiterSCF) {
    if (nStepSCF != 0) {
      calculateN(_gamma_scf);
      calculateV(_gamma_scf);
    }
    for (int k = 0; k < _o.size(); ++k) {
      icl[k] = -1;

      ldouble lmain_eq = _o[k]->l();
      int lmain = _o[k]->l();
      int mmain = _o[k]->m();
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
  }
}

void DFT::calculateN(ldouble gamma) {
  std::cout << "Calculating n." << std::endl;

  _nsum_up = std::vector<ldouble>(_g->N(), 0);
  _nsum_dw = std::vector<ldouble>(_g->N(), 0);

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();

    std::cout << "Calculating n term from k = " << k1 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

    for (int k = 0; k < _g->N(); ++k) {
      for (int ml_idx = 0; ml_idx < _o[k1]->term().size(); ++ml_idx) {
        int ml = ml_idx/2 - l1;
        if (_o[k1]->term()[ml_idx] == '+')
          _nsum_up[k] += std::pow(_o[k1]->getNorm(k, *_g), 2);
        else if (_o[k1]->term()[ml_idx] == '-')
          _nsum_dw[k] += std::pow(_o[k1]->getNorm(k, *_g), 2);
      }
    }
  }

  for (int k = 0; k < _g->N(); ++k) _n_up[k] = (1-gamma)*_n_up[k] + gamma*_nsum_up[k];
  for (int k = 0; k < _g->N(); ++k) _n_dw[k] = (1-gamma)*_n_dw[k] + gamma*_nsum_dw[k];

}


void DFT::calculateV(ldouble gamma) {
  std::cout << "Calculating u." << std::endl;
  _u = std::vector<ldouble>(_g->N(), 0);

  ldouble Q = 0;
  std::vector<ldouble> E(_g->N(), 0); // electric field
  for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
    ldouble r1 = (*_g)(ir1);
    ldouble dr = 0;
    if (ir1 < _g->N()-1) dr = (*_g)(ir1+1) - (*_g)(ir1);
    ldouble n_up = 0;
    ldouble n_dw = 0;
    n_up += _n_up[ir1];
    n_dw += _n_dw[ir1];
    Q += (n_up + n_dw)*std::pow(r1, 2)*dr;
    E[ir1] = Q/std::pow(r1, 2);
  }
  _u[_g->N()-1] = Q/(*_g)(_g->N()-1);
  for (int ir1 = _g->N()-2; ir1 >= 0; --ir1) {
    ldouble dr = (*_g)(ir1+1) - (*_g)(ir1);
    _u[ir1] = _u[ir1+1] + E[ir1]*dr;
  }

  std::cout << "Calculating vex using LDA." << std::endl;
  _vex_lda_up = std::vector<ldouble>(_g->N(), 0);
  _vex_lda_dw = std::vector<ldouble>(_g->N(), 0);
  // Ex = -3/4 (3/pi)^(1/3) int n^(4/3) dr = int exc n dr
  // exc = -3/4 (3/pi)^(1/3) n^(1/3)
  //
  // vx = dE/dn
  // with spin:
  // Ex = 0.5*(Ex[2*n_up] + Ex[2*n_dw])
  // vx = 0.5*(2*dE[n_up]/dn_up + 2*dE[n_dw]/dn_dw) = dE/dn_up + dE/dn_dw
  // vx = exc + n dexc/dn

  // rs = (3/(4*pi*n))^(1/3)
  // ex = -3/(4pi)*(3 pi^2 n)^(1/3) = -3/4 (3/pi)^(1/3) n^(1/3)
  ldouble Ax = -3.0/4.0*std::pow(3.0/M_PI, 1.0/3.0);
  for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
    //_vex_lda_up[ir1] += Ax*std::pow(_n_up[ir1] + _n_dw[ir1], 1.0/3.0);
    //_vex_lda_dw[ir1] += Ax*std::pow(_n_up[ir1] + _n_dw[ir1], 1.0/3.0);

    //if (_n_up[ir1] + _n_dw[ir1] != 0) {
    //  _vex_lda_up[ir1] += (_n_up[ir1])*Ax*std::pow(_n_up[ir1] + _n_dw[ir1], -2.0/3.0);
    //  _vex_lda_dw[ir1] += (_n_dw[ir1])*Ax*std::pow(_n_up[ir1] + _n_dw[ir1], -2.0/3.0);
    //}

    _vex_lda_up[ir1] += Ax*4.0/3.0*std::pow(_n_up[ir1], 1.0/3.0);
    _vex_lda_dw[ir1] += Ax*4.0/3.0*std::pow(_n_dw[ir1], 1.0/3.0);

    //if (_n_up[ir1] + _n_dw[ir1] != 0) {
    //  _vex_lda_up[ir1] += (_n_up[ir1] + _n_dw[ir1])*Ax*std::pow(2*_n_up[ir1], -2.0/3.0);
    //  _vex_lda_dw[ir1] += (_n_up[ir1] + _n_dw[ir1])*Ax*std::pow(2*_n_dw[ir1], -2.0/3.0);
    //}
  }

  std::cout << "Calculating sum of SCF potentials." << std::endl;
  for (int k = 0; k < _g->N(); ++k) _vsum_up[k] = (1-gamma)*_vsum_up[k] + gamma*(_u[k] + _vex_lda_up[k]);
  for (int k = 0; k < _g->N(); ++k) _vsum_dw[k] = (1-gamma)*_vsum_dw[k] + gamma*(_u[k] + _vex_lda_dw[k]);
}


void DFT::calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E) {
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
      int s1 = _om.s(idx1);

      for (int idx2 = 0; idx2 < N; ++idx2) {
        int k2 = _om.orbital(idx2);
        int l2 = _om.l(idx2);
        ldouble l2_eq = _om.l(idx2);
        int m2 = _om.m(idx2);

        if (idx1 == idx2) {
          ldouble a = 0;
          if (s1 > 0) {
            if (_g->isLog()) a = 2*std::pow(r, 2)*(E[k1] - _pot[i] - _vsum_up[i]) - std::pow(l1_eq + 0.5, 2);
            else a = 2*(E[k1] - _pot[i] - _vsum_up[i] - l1_eq*(l1_eq + 1)/std::pow((*_g)(i), 2));
          } else {
            if (_g->isLog()) a = 2*std::pow(r, 2)*(E[k1] - _pot[i] - _vsum_dw[i]) - std::pow(l1_eq + 0.5, 2);
            else a = 2*(E[k1] - _pot[i] - _vsum_dw[i] - l1_eq*(l1_eq + 1)/std::pow((*_g)(i), 2));
          }

          F[i](idx1,idx1) += 1 + a*std::pow(_g->dx(), 2)/12.0;
          Lambda[i](idx1,idx1) += 1 + a*std::pow(_g->dx(), 2)/12.0;
        }
      }
    }
    K[i] = F[i].inverse();
    //for (int idxD = 0; idxD < N; ++idxD) Lambda[i](idxD, idxD) = 1.0/Lambda[i](idxD, idxD);
    //K[i] = Lambda[i]*K[i];
    //K[i] = (MatrixXld::Identity(N,N) + std::pow(_g->dx(), 2)/12.0*K[i] + std::pow(_g->dx(), 4)/144.0*(K[i]*K[i]))*Lambda[i];
  }
}

void DFT::addOrbital(Orbital *o) {
  o->N(_g->N());
  _o.push_back(o);
  // initialise energies and first solution guess
  for (int k = 0; k < _o.size(); ++k) {
    _o[k]->E(-_Z*_Z*0.5/std::pow(_o[k]->n(), 2));

    for (int ir = 0; ir < _g->N(); ++ir) { // for each radial point
      (*_o[k])(ir) = std::pow(_Z*(*_g)(ir)/((ldouble) _o[k]->n()), _o[k]->l()+0.5)*std::exp(-_Z*(*_g)(ir)/((ldouble) _o[k]->n()));
    }
  }
  _n_up = std::vector<ldouble>(_g->N(), 0);
  _n_dw = std::vector<ldouble>(_g->N(), 0);
  _u = std::vector<ldouble>(_g->N(), 0);
  _vex_lda_up = std::vector<ldouble>(_g->N(), 0);
  _vex_lda_dw = std::vector<ldouble>(_g->N(), 0);
  _vsum_up = std::vector<ldouble>(_g->N(), 0);
  _vsum_dw = std::vector<ldouble>(_g->N(), 0);
}




