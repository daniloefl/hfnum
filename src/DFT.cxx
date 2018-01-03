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
  : SCF() {
  load(fname);
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
  f << std::setw(10) << "central" << " " << std::setw(10) << _central << std::endl;
  f << std::setw(10) << "grid.isLog" << " " << std::setw(10) << _g->isLog() << std::endl;
  f << std::setw(10) << "grid.dx" << " " << std::setw(10) << _g->dx() << std::endl;
  f << std::setw(10) << "grid.N" << " " << std::setw(10) << _g->N() << std::endl;
  f << std::setw(10) << "grid.rmin" << " " << std::setw(10) << (*_g)(0) << std::endl;
  for (int i = 0; i < _o.size(); ++i) {
    f << std::setw(10) << "orbital" << " " << std::setw(10) << i;
    f << " " << std::setw(5) << "n" << " " << std::setw(5) << _o[i]->initialN();
    f << " " << std::setw(5) << "l" << " " << std::setw(5) << _o[i]->initialL();
    f << " " << std::setw(5) << "m" << " " << std::setw(5) << _o[i]->initialM();
    f << " " << std::setw(5) << "s" << " " << std::setw(5) << _o[i]->spin();
    f << " " << std::setw(5) << "E" << " " << std::setw(64) << std::setprecision(60) << _o[i]->E();
    f << " " << std::setw(10) << "sph_size" << " " << std::setw(5) << _o[i]->getSphHarm().size();
    for (int idx = 0; idx < _o[i]->getSphHarm().size(); ++idx) {
      f << " " << std::setw(5) << "sph_l" << " " << std::setw(5) << _o[i]->getSphHarm()[idx].first;
      f << " " << std::setw(5) << "sph_m" << " " << std::setw(5) << _o[i]->getSphHarm()[idx].second;
      f << " " << std::setw(5) << "value";
      for (int ir = 0; ir < _g->N(); ++ir) {
        const ldouble v = ((const Orbital) (*_o[i]))(ir, _o[i]->getSphHarm()[idx].first, _o[i]->getSphHarm()[idx].second);
        f << " " << std::setw(64) << std::setprecision(60) << v;
      }
    }
    f << std::endl;
  }
  //for (auto &vr : _n_up) {
  //  const std::pair<int, int> &lm = vr.first;
  //  const std::vector<ldouble> &vradial = vr.second;
    f << std::setw(10) << "n_up";
  //  f << " " << std::setw(5) << "l" << " " << std::setw(5) << lm.first;
  //  f << " " << std::setw(5) << "m" << " " << std::setw(5) << lm.second;
    f << " " << std::setw(5) << "value";
  //  for (int ir = 0; ir < vradial.size(); ++ir) {
  //    f << " " << std::setw(64) << std::setprecision(60) << vradial[ir];
    for (int ir = 0; ir < _n_up.size(); ++ir) {
      f << " " << std::setw(64) << std::setprecision(60) << _n_up[ir];
    }
    f << std::endl;
  //}
  //for (auto &vr : _n_dw) {
  //  const std::pair<int, int> &lm = vr.first;
  //  const std::vector<ldouble> &vradial = vr.second;
    f << std::setw(10) << "n_dw";
  //  f << " " << std::setw(5) << "l" << " " << std::setw(5) << lm.first;
  //  f << " " << std::setw(5) << "m" << " " << std::setw(5) << lm.second;
    f << " " << std::setw(5) << "value";
  //  for (int ir = 0; ir < vradial.size(); ++ir) {
  //    f << " " << std::setw(64) << std::setprecision(60) << vradial[ir];
    for (int ir = 0; ir < _n_dw.size(); ++ir) {
      f << " " << std::setw(64) << std::setprecision(60) << _n_dw[ir];
    }
    f << std::endl;
  //}
}

void DFT::load(const std::string fin) {
  std::ifstream f(fin.c_str());
  std::string line;

  bool g_isLog = true;
  ldouble g_dx = 1e-1;
  int g_N = 220;
  ldouble g_rmin = 1e-6;

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
      ss >> _central;
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

      int o_N, o_L, o_M, o_S;
      ss >> trash >> o_N >> trash >> o_L >> trash >> o_M >> trash >> o_S;

      ldouble o_E;
      ss >> trash >> o_E;


      _owned_orb.push_back(new Orbital(o_S, o_N, o_L, o_M));
      _o.push_back(_owned_orb[_owned_orb.size()-1]);
      int k = _o.size()-1;
      _o[k]->N(g_N);
      _o[k]->E(o_E);
      int sphSize = 1;
      ss >> trash >> sphSize;
      for (int idx = 0; idx < sphSize; ++idx) {
        int l;
        int m;
        ss >> trash >> l >> trash >> m;

        if (l != o_L && m != o_M) _o[k]->addSphHarm(l, m);

        ss >> trash;

        ldouble read_value;
        for (int ir = 0; ir < g_N; ++ir) { // for each radial point
          ss >> read_value;
          (*_o[k])(ir, l, m) = read_value;
        }
      }
    } else if (mode == "n_up") {
      std::string trash;
      int v_l, v_m;
      //ss >> trash >> v_l >> trash >> v_m;

      ss >> trash;

      //_n_up[std::pair<int, int>(v_l, v_m)] = std::vector<ldouble>(g_N, 0);
      _n_up = std::vector<ldouble>(g_N, 0);
      ldouble read_value;
      for (int k = 0; k < g_N; ++k) {
        ss >> read_value;
        _n_up[k] = read_value;
      }
    } else if (mode == "n_dw") {
      std::string trash;

      int v_l, v_m;
      //ss >> trash >> v_l >> trash >> v_m;

      ss >> trash;

      //_n_dw[std::pair<int, int>(v_l, v_m)] = std::vector<ldouble>(g_N, 0);
      _n_dw = std::vector<ldouble>(g_N, 0);
      ldouble read_value;
      for (int k = 0; k < g_N; ++k) {
        ss >> read_value;
        _n_dw[k] = read_value;
      }
    }
  }
  std::cout << "Load resetting grid with isLog = " << g_isLog << ", dx = " << g_dx << ", g_N = " << g_N << ", g_rmin = " << g_rmin << std::endl;
  _g->reset(g_isLog, g_dx, g_N, g_rmin);
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

ldouble DFT::solveForFixedPotentials(int Niter, ldouble F0stop) {
  ldouble gamma = 1; // move in the direction of the negative slope with this velocity per step

  std::string strMethod = "";
  if (_method == 0) {
    strMethod = "Sparse Matrix Numerov";
  } else if (_method == 1) {
    strMethod = "Iterative Numerov with Gordon method for initial condition (http://aip.scitation.org/doi/pdf/10.1063/1.436421)";
  } else if (_method == 2) {
    strMethod = "Iterative Renormalised Numerov (http://aip.scitation.org/doi/pdf/10.1063/1.436421)";
  }

  ldouble F = 0;
  bool allNodesOk = false;
  do {
    int nStep = 0;
    while (nStep < Niter) {
      gamma = 0.5*(1 - std::exp(-(nStep+1)/5.0));
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
        ldouble newE = (_o[k]->E()+stepdE);
        //if (newE > _Emax[k]) newE = 0.5*(_Emax[k] + _Emin[k]);
        //if (newE < _Emin[k]) newE = 0.5*(_Emax[k] + _Emin[k]);
        std::cout << std::setw(5) << k << " " << std::setw(16) << std::setprecision(12) << _o[k]->E() << " " << std::setw(16) << std::setprecision(12) << newE << " " << std::setw(16) << std::setprecision(12) << _Emin[k] << " " << std::setw(16) << std::setprecision(12) << _Emax[k] << " " << std::setw(5) << _nodes[k] << std::endl;
        _o[k]->E(newE);
      }

      if (std::fabs(*std::max_element(_dE.begin(), _dE.end(), [](ldouble a, ldouble b) -> bool { return std::fabs(a) < std::fabs(b); } )) < F0stop) break;
      //if (std::fabs(F) < F0stop) break;
    }

    allNodesOk = true;
    //for (int k = 0; k < _o.size(); ++k) {
    //  if (_nodes[k] > _o[k]->initialN() - _o[k]->initialL() - 1) {
    //    allNodesOk = false;
    //    std::cout << "Found too many nodes in orbital " << k << ": I will try again starting at a lower energy." << std::endl;
    //    _Emax[k] = _o[k]->E();
    //    _o[k]->E(0.5*(_Emax[k] + _Emin[k]));
    //  } else if (_nodes[k] < _o[k]->initialN() - _o[k]->initialL() - 1) {
    //    allNodesOk = false;
    //    std::cout << "Found too few nodes in orbital " << k << ": I will try again starting at a higher energy." << std::endl;
    //    _Emin[k] = _o[k]->E();
    //    _o[k]->E(0.5*(_Emax[k] + _Emin[k]));
    //  }
    //}

  } while (!allNodesOk);
  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble DFT::stepGordon(ldouble gamma) {
  int N = 0;
  for (int k = 0; k < _o.size(); ++k) {
    N += _o[k]->getSphHarm().size();
  }

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = -1e-3;
    E[k] = _o[k]->E();
    l[k] = _o[k]->initialL();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, E);

  ldouble Fn = _igs.solve(E, l, Fmn, Kmn, matched);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->initialL();
    int m = _o[k]->initialM();
    int idx = _om.index(k, l, m);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i, l, m) = matched[i](idx);
      if (i >= 10 && (*_g)(i) < std::pow(_o.size(),2) && i < _g->N() - 4 && matched[i](idx)*matched[i-1](idx) <= 0) {
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
ldouble DFT::stepRenormalised(ldouble gamma) {
  int N = _om.N();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = 1e-3;
    E[k] = _o[k]->E();
    l[k] = _o[k]->initialL();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, E);

  ldouble Fn = _irs.solve(E, l, Fmn, Kmn, matched);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->initialL();
    int m = _o[k]->initialM();
    int idx = _om.index(k, l, m);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i, l, m) = matched[i](idx);
      if (i >= 10 && i < _g->N() - 4 && matched[i](idx)*matched[i-1](idx) <= 0) {
        //ldouble deriv = (matched[i](idx) - matched[i-1](idx))/(_g(i) - _g(i-1));
        //if (std::fabs(deriv) > 1e-2) {
          _nodes[k] += 1;
          std::cout << "Orbital " << k << ": Found node at i=" << i << ", r = " << (*_g)(i) << std::endl;
        //}
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

  ldouble F = Fn;
  for (int k = 0; k < _o.size(); ++k) {
    //_dE[k] = gamma*dX(k); // to use the curvature for extrema finding
    if (grad(k) != 0) {
      //_dE[k] = -gamma*Fn/grad(k); // for root finding
      _dE[k] = Fn/grad(k); // for root finding
      _dE[k] *= -gamma;
    } else {
      _dE[k] = 0;
    }
    //if (std::fabs(_dE[k]) > 0.1) _dE[k] = 0.1*_dE[k]/std::fabs(_dE[k]);
    std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble DFT::stepSparse(ldouble gamma) {
  // 1) build sparse matrix _A
  // 2) build sparse matrix _b
  _lsb.prepareMatrices(_A, _b0, _pot, _vsum_up, _vsum_dw);
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
  _lsb.propagate(_b, _dE, gamma);
  // 5) change results in _dE[k]

  // count nodes for monitoring
  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->initialL();
    int m = _o[k]->initialM();
    for (int i = 0; i < _g->N(); ++i) {
      if (i >= 10 && (*_g)(i) < std::pow(_o.size(),2) && i < _g->N() - 4 && (*_o[k])(i, l, m)*(*_o[k])(i-1, l, m) <= 0) {
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


ldouble DFT::getE0() {
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    E0 += _o[k]->E();
  }
  ldouble J = 0;
  for (int ir = 0; ir < _g->N(); ++ir) {
    ldouble r = (*_g)(ir);
    ldouble dr = 0;
    if (ir < _g->N()-1)
      dr = (*_g)(ir+1) - (*_g)(ir);
    J += _u[ir]*(_n_up[ir] + _n_dw[ir])*std::pow(r, 2)*dr;
  }
  E0 += -0.5*J;
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
    calculateV(_gamma_scf);
  }
}

void DFT::calculateN(ldouble gamma) {
  std::cout << "Calculating n." << std::endl;

  _nsum_up = std::vector<ldouble>(_g->N(), 0);
  _nsum_dw = std::vector<ldouble>(_g->N(), 0);

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->initialL();
    int m1 = _o[k1]->initialM();
    int s1 = _o[k1]->spin();

    std::cout << "Calculating n term from k = " << k1 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

    for (int k = 0; k < _g->N(); ++k) {
      if (s1 > 0) {
        _nsum_up[k] += std::pow(_o[k1]->getNorm(k, l1, m1, *_g), 2);
      } else {
        _nsum_dw[k] += std::pow(_o[k1]->getNorm(k, l1, m1, *_g), 2);
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

    _vex_lda_up[ir1] += 0.5*Ax*std::pow(2*_n_up[ir1], 1.0/3.0);
    _vex_lda_dw[ir1] += 0.5*Ax*std::pow(2*_n_dw[ir1], 1.0/3.0);

    if (_n_up[ir1] + _n_dw[ir1] != 0) {
      _vex_lda_up[ir1] += 0.5*(2*_n_up[ir1])*Ax*std::pow(2*_n_up[ir1], -2.0/3.0);
      _vex_lda_dw[ir1] += 0.5*(2*_n_dw[ir1])*Ax*std::pow(2*_n_dw[ir1], -2.0/3.0);
    }
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
    _o[k]->E(-_Z*_Z*0.5/std::pow(_o[k]->initialN(), 2));

    for (int idx = 0; idx < _o[k]->getSphHarm().size(); ++idx) {
      int l = _o[k]->getSphHarm()[idx].first;
      int m = _o[k]->getSphHarm()[idx].second;
      for (int ir = 0; ir < _g->N(); ++ir) { // for each radial point
        (*_o[k])(ir, l, m) = std::pow(_Z*(*_g)(ir)/((ldouble) _o[k]->initialN()), l+0.5)*std::exp(-_Z*(*_g)(ir)/((ldouble) _o[k]->initialN()));
      }
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




