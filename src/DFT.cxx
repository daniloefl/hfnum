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
}

DFT::DFT(const std::string fname)
  : SCF(fname) {
}

DFT::~DFT() {
}

ldouble DFT::getE0() {
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

void DFT::solve(int NiterSCF, int Niter, ldouble F0stop) {
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
    calculateN(_gamma_scf);
    calculateVd(_gamma_scf);
  }
}

void DFT::calculateN(ldouble gamma) {
  std::cout << "Calculating n." << std::endl;

  _nsum_up.clear();
  _nsum_up[std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
  _nsum_up[std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
  _nsum_up[std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
  _nsum_up[std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);

  _nsum_dw.clear();
  _nsum_dw[std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
  _nsum_dw[std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
  _nsum_dw[std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
  _nsum_dw[std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->initialL();
    int m1 = _o[k1]->initialM();
    int s1 = _o[k1]->spin();

    std::cout << "Calculating n term from k = " << k1 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

    for (int k = 0; k < _g->N(); ++k) {
      if (s1 > 0) {
        _nsum_up[std::pair<int,int>(l1, m1)][k] += std::pow(_o[k1]->getNorm(k, l1, m1, *_g), 2);
      } else {
        _nsum_dw[std::pair<int,int>(l1, m1)][k] += std::pow(_o[k1]->getNorm(k, l1, m1, *_g), 2);
      }
    }
  }

  for (auto &i : _nsum_up) {
    const std::pair<int, int> k = i.first;
    int lj = k.first;
    int mj = k.second;
    std::vector<ldouble> &currentN = _n_up[std::pair<int,int>(lj, mj)];
    for (int k = 0; k < _g->N(); ++k) currentN[k] = (1-gamma)*currentN[k] + gamma*_nsum_up[std::pair<int,int>(lj,mj)][k];
  }

  for (auto &i : _nsum_dw) {
    const std::pair<int, int> k = i.first;
    int lj = k.first;
    int mj = k.second;
    std::vector<ldouble> &currentN = _n_dw[std::pair<int,int>(lj, mj)];
    for (int k = 0; k < _g->N(); ++k) currentN[k] = (1-gamma)*currentN[k] + gamma*_nsum_dw[std::pair<int,int>(lj,mj)][k];
  }

  for (int k = 0; k < _g->N(); ++k) {
    _u[k] = 0;
  }

  ldouble Q = 0;
  std::vector<ldouble> E(_g->N(), 0); // electric field
  for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
    ldouble r1 = (*_g)(ir1);
    ldouble dr = 0;
    if (ir1 < _g->N()-1) dr = (*_g)(ir1+1) - (*_g)(ir1);
    ldouble n_up = 0;
    ldouble n_dw = 0;
    for (auto &i : _n_up) { // ignores spherical components
      const std::pair<int, int> k = i.first;
      int lj = k.first;
      int mj = k.second;
      n_up += _n_up[k][ir1];
      n_dw += _n_dw[k][ir1];
    }
    Q += (n_up + n_dw)*std::pow(r1, 2)*dr;
    E[ir1] = Q/std::pow(r1, 2);
  }
  _u[_g->N()-1] = Q/(*_g)(_g->N()-1);
  for (int ir1 = _g->N()-2; ir1 >= 0; --ir1) {
    ldouble dr = (*_g)(ir1+1) - (*_g)(ir1);
    _u[ir1] = _u[ir1+1] + E[ir1]*dr;
  }

}


void DFT::calculateVd(ldouble gamma) {
  std::cout << "Calculating Vd." << std::endl;

  for (int ko = 0; ko < _o.size(); ++ko) {
    for (auto &idx : _vd[ko]) {
      int lj = idx.first.first;
      int mj = idx.first.second;
      std::cout << "Adding SCF potential term for eq. " << ko << std::endl;
      std::vector<ldouble> &currentVd = _vd[ko][std::pair<int,int>(lj, mj)];
      for (int k = 0; k < _g->N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*_u[k];
    }
  }
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
    _o[k]->E(-_Z*_Z*0.5/std::pow(_o[k]->initialN(), 2));

    for (int idx = 0; idx < _o[k]->getSphHarm().size(); ++idx) {
      int l = _o[k]->getSphHarm()[idx].first;
      int m = _o[k]->getSphHarm()[idx].second;
      for (int ir = 0; ir < _g->N(); ++ir) { // for each radial point
        (*_o[k])(ir, l, m) = std::pow(_Z*(*_g)(ir)/((ldouble) _o[k]->initialN()), l+0.5)*std::exp(-_Z*(*_g)(ir)/((ldouble) _o[k]->initialN()));
      }
    }
  }
  _n_up.clear();
  _n_dw.clear();
  _vd.clear();
  _n_up = Vd();
  _n_up[std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
  _n_up[std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
  _n_up[std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
  _n_up[std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
  _n_dw = Vd();
  _n_dw[std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
  _n_dw[std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
  _n_dw[std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
  _n_dw[std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    _vd[k] = Vd();
    _vd[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vd[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
    _vd[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vd[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
  }

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vd();
    _vdsum[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
  }
}




