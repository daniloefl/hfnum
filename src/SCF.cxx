#include "SCF.h"
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

#include <boost/python.hpp>

#include <Python.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <cstdlib>

#include "HFException.h"

using namespace boost;

SCF::SCF(ldouble Z)
  : _g(new Grid(expGrid, 1.0/32.0, (int) ((std::log(12.0) + 5 + std::log(Z))/(1.0/32.0))+1, std::exp(-5)/Z)),
  _Z(Z), _om(*_g, _o), _lsb(*_g, _o, icl, _om), _iss(*_g, _o, icl, _om) {
  _findRoots = true;
  _own_grid = true;
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
  _gamma_scf = 0.5;
  _method = 1;
  _isSpinDependent = false;
}

Grid &SCF::getGrid() {
  return *_g;
}

python::list SCF::getR() const {
  return _g->getR();
}

void SCF::resetGrid(int t, ldouble dx, int N, ldouble rmin) {
  _g->reset((gridType) t, dx, N, rmin);
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
}

void SCF::setZ(ldouble Z) {
  _Z = Z;
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
  _iss.setZ(_Z);
}

ldouble SCF::Z() {
  return _Z;
}

SCF::~SCF() {
  for (auto &o : _owned_orb) {
    delete o;
  }
  _owned_orb.clear();
  if (_own_grid) delete _g;
}

int SCF::getNOrbitals() {
  return _o.size();
}

int SCF::getOrbital_n(int no) {
  return _o[no]->n();
}

std::string SCF::getOrbitalName(int no) {
  std::string name = "";
  name += std::to_string(_o[no]->n());
  int l = _o[no]->l();
  int m = _o[no]->m();
  int s = _o[no]->spin();
  int g = _o[no]->g();
  if (l == 0) name += "s";
  else if (l == 1) name += "p";
  else if (l == 2) name += "d";
  else if (l == 3) name += "f";
  else if (l == 4) name += "g";
  else if (l == 5) name += "h";
  else name += "?";
  if (s == 0) {
    name += "^";
    name += std::to_string(g);
  } else {
    name += "_{m=";
    name += std::to_string(m);
    name += "}";
    if (s > 0)
      name += "^+";
    else
      name += "^-";
  }
  return name;
}

ldouble SCF::getOrbital_E(int no) {
  return _o[no]->E();
}

int SCF::getOrbital_l(int no) {
  return _o[no]->l();
}

int SCF::getOrbital_m(int no) {
  return _o[no]->m();
}

int SCF::getOrbital_s(int no) {
  return _o[no]->spin();
}

void SCF::method(int m) {
  if (m < 0) m = 0;
  if (m > 1) m = 1;
  _method = m;
}

python::list SCF::getNucleusPotentialPython() {
  python::list l;
  std::vector<ldouble> v = getNucleusPotential();
  for (int k = 0; k < _g->N(); ++k) l.append(v[k]);
  return l;
}
std::vector<ldouble> SCF::getOrbital(int no) {
  Orbital *o = _o[no];
  std::vector<ldouble> res;
  for (int k = 0; k < _g->N(); ++k) {
    res.push_back(o->getNorm(k, *_g));
  }
  return res;
}

std::vector<ldouble> SCF::getOrbitalCentral(int no) {
  Orbital *o = _o[no];
  std::vector<ldouble> res;
  for (int k = 0; k < _g->N(); ++k) {
    res.push_back(o->getNorm(k, *_g));
  }
  return res;
}

python::list SCF::getOrbitalCentralPython(int no) {
  python::list l;
  std::vector<ldouble> v = getOrbitalCentral(no);
  for (int k = 0; k < _g->N(); ++k) l.append(v[k]);
  return l;
}

void SCF::addOrbitalPython(python::object o) {
  Orbital *orb = python::extract<Orbital *>(o);
  addOrbital(orb);
}


void SCF::gammaSCF(ldouble g) {
  _gamma_scf = g;
}


std::vector<ldouble> SCF::getNucleusPotential() {
  return _pot;
}

ldouble SCF::solveForFixedPotentials(int Niter, ldouble F0stop) {
  ldouble gamma = 1; // move in the direction of the negative slope with this velocity per step

  std::string strMethod = "";
  if (_method == 0) {
    strMethod = "Sparse Matrix Numerov";
  } else if (_method == 1) {
    strMethod = "Iterative Standard Numerov with non-homogeneous term";
  }

  _historyL.clear();
  _historyE.clear();
  _historyF.clear();

  // create lambda map to flatten it
  // and resize it to be sure it has the proper size
  _lambdaMap.clear();
  int lcount = 0;
  if (!_isSpinDependent) {
    for (int i = 0; i < _o.size(); ++i) {
      for (int j = i+1; j < _o.size(); ++j) {
        if (_o[i]->l() != _o[j]->l()) continue;
        _lambdaMap[100*i + j] = lcount;
        _lambdaMap[100*j + i] = lcount;
        lcount++;
      }
    }
  }
  _lambda.resize(lcount, 0);
  _dlambda.resize(lcount, 0);

  ldouble Lstop = 1e-3;

  ldouble F = 0;
  int nStep = 0;
  _reestimateHessian = true;
  while (nStep < Niter) {
    // compute sum of squares of F(x_old)
    nStep += 1;
    if (_method == 0) {
      gamma = 0.5*(1 - std::exp(-(nStep+1)/20.0));
      F = stepSparse(gamma);
    } else if (_method == 1) {
      gamma = 1; //0.5*(1 - std::exp(-(nStep+1)/5.0));
      if (_findRoots) {
        F = stepStandard(gamma);
      } else {
        F = stepStandardMinim(gamma);
      }
    }
  
    // change orbital energies
    std::cout << "Orbital energies at step " << nStep << ", with constraint = " << std::setw(16) << F << ", method = " << strMethod << "." << std::endl;
    std::cout << std::setw(5) << std::right << "Index" << " "
              << std::setw(16) << std::right << "Name" << " "
              << std::setw(16) << std::right << "Energy (H)" << " "
              << std::setw(16) << std::right << "Step (H)" << " "
              << std::setw(16) << std::right << "Min. (H)" << " "
              << std::setw(16) << std::right << "Max. (H)" << " "
              << std::setw(5) << std::right << "nodes" << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      ldouble stepdE = _dE[k];
      ldouble newE = (_o[k]->E()+stepdE);
      std::cout << std::setw(5) << std::right << k << " "
                << std::setw(16) << std::right << getOrbitalName(k) << " "
                << std::setw(16) << std::right << std::setprecision(10) << _o[k]->E() << " "
                << std::setw(16) << std::right << std::setprecision(10) << stepdE << " "
                << std::setw(16) << std::right << std::setprecision(10) << _Emin[k] << " "
                << std::setw(16) << std::right << std::setprecision(10) << _Emax[k] << " "
                << std::setw(5) << std::right << _nodes[k] << std::endl;
      _o[k]->E(newE);
    }
    std::cout << "Lagrange multipliers" << std::endl;
    std::cout << std::setw(10) << std::right << "Variable" << " "
              << std::setw(16) << std::right << "Value" << " "
              << std::setw(16) << std::right << "Step" << std::endl;
    for (auto &k : _lambdaMap) {
      int k2 = k.first / 100;
      int k1 = k.first % 100;
      if (k1 > k2) continue;
      std::string s = "\\lambda_{";
      s += std::to_string(k1);
      s += ",";
      s += std::to_string(k2);
      s += "}";
      std::cout << std::setw(10) << s << " "
                << std::setw(16) << std::setprecision(10) << std::right << _lambda[k.second] << " "
                << std::setw(16) << std::setprecision(10) << std::right << _dlambda[k.second] << std::endl;
    }
    std::cout.unsetf(std::cout.showpos);
    for (int k = 0; k < _lambda.size(); ++k) {
      ldouble stepdE = _dlambda[k];
      ldouble newE = (_lambda[k]+stepdE);
      _lambda[k] = newE;
    }
  
    std::vector<ldouble> dErel;
    for (int k = 0; k < _dE.size(); ++k) {
      dErel.push_back(_dE[k]/_o[k]->E());
    }
    bool stop = std::fabs(*std::max_element(dErel.begin(), dErel.end(), [](ldouble a, ldouble b) -> bool { return std::fabs(a) < std::fabs(b); } )) < F0stop;
    if (_dlambda.size() > 0) {
      stop = stop && (std::fabs(*std::max_element(_dlambda.begin(), _dlambda.end(), [](ldouble a, ldouble b) -> bool { return std::fabs(a) < std::fabs(b); } )) < Lstop);
    }
    if (stop) break;

    // check for interrupt signal
    PyErr_CheckSignals();
    if (PyErr_Occurred())
      throw HFException("Received signal. Interrupting.");

  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepSparse(ldouble gamma) {
  // 1) build sparse matrix _A
  // 2) build sparse matrix _b
  if (_isSpinDependent) {
    _lsb.prepareMatrices(_A, _b0, _pot, _vsum_up, _vsum_dw);
  } else {
    _lsb.prepareMatrices(_A, _b0, _pot, _vd, _vex, _lambda, _lambdaMap);
  }
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
    int l = _o[k]->l();
    int m = _o[k]->m();

    ldouble a_max = 0;
    for (int i = 0; i < _g->N(); ++i) {
      if (std::fabs((*_o[k])(i)) > a_max) {
        a_max = std::fabs((*_o[k])(i));
      }
    }
    std::vector<ldouble> alist;
    for (int i = 0; i < _g->N(); ++i) {
      if (std::fabs((*_o[k])(i)) < 0.01*a_max) continue;
      alist.push_back((*_o[k])(i));
    }
    for (int i = 0; i < alist.size(); ++i) {
      if (i >= 1 && alist[i]*alist[i-1] < 0) {
        _nodes[k] += 1;
      }
    }
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "Orbital " << k << ", dE = " << _dE[k] << std::endl;
    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      std::cout << "Orbital " << k << ", new dE = " << _dE[k] << std::endl;
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      std::cout << "Orbital " << k << ", new dE = " << _dE[k] << std::endl;
    } else {
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }
    }
  }

  // 6) calculate F = sum _b[k]^2
  ldouble F = 0;
  for (int k = 0; k < _b.rows(); ++k) F += std::pow(_b(k), 2);
  return F;
}

ldouble SCF::solveStandard(VectorXld &E, VectorXld &lambda, VectorXld &Sn, VectorXld &Fn, std::map<int, Vradial> &matchedOrb) {
  Sn.resize(lambda.size());
  Sn.setZero();

  if (_isSpinDependent) {
    Fn = _iss.solve(E, _pot, _vsum_up, _vsum_dw, lambda, _lambdaMap, matchedOrb);
  } else {
    Fn = _iss.solve(E, _pot, _vd, _vex, lambda, _lambdaMap, matchedOrb);
  }

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (k1 <= k2) continue;
      if (_o[k1]->l() != _o[k2]->l()) continue;
      int lidx = _lambdaMap[k1*100+k2];
      for (int ir = 0; ir < _g->N()-1; ++ir) {
        if (_g->isLog()) // y = psi * sqrt(r) and dr = r dx, so psi1 * psi2 * r^2 * dr = y1 * y2 / r * r^2 * r * dx = y1 y2 r^2 dx
          Sn(lidx) += matchedOrb[k1][ir]*matchedOrb[k2][ir]*std::pow((*_g)(ir), 2)*_g->dx();
        else if (_g->isLin()) // y = psi and dr = dx, so psi1 psi2 r^2 dr = y1 y2 r^2 dx
          Sn(lidx) += matchedOrb[k1][ir]*matchedOrb[k2][ir]*std::pow((*_g)(ir), 2)*_g->dx();
      }
    }
  }
  ldouble Fn2 = 0;
  for (int k = 0; k < E.size(); ++k) Fn2 += std::pow(Fn(k), 2);
  for (int k = 0; k < lambda.size(); ++k) Fn2 += std::pow(Sn(k), 2);
  return Fn2;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepStandard(ldouble gamma) {
  int N = _om.N();

  VectorXld E(_o.size());
  for (int k = 0; k < _o.size(); ++k) E(k) = _o[k]->E();
  VectorXld L(_lambda.size());
  for (int k = 0; k < _lambda.size(); ++k) L(k) = _lambda[k];

  VectorXld Fn(E.size());
  VectorXld Sn(L.size());
  ldouble sumFn2 = solveStandard(E, L, Sn, Fn, matchedSt);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);

    ldouble a_max = 0;
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matchedSt[idx][i];
      if (std::fabs((*_o[k])(i)) > a_max) {
        a_max = std::fabs((*_o[k])(i));
      }
    }
    std::vector<ldouble> alist;
    std::vector<ldouble> alist_i;
    for (int i = 0; i < _g->N(); ++i) {
      if (std::fabs((*_o[k])(i)) < 0.01*a_max) continue;
      alist.push_back((*_o[k])(i));
      alist_i.push_back(i);
    }
    for (int i = 0; i < alist.size(); ++i) {
      if (i >= 1 && alist[i]*alist[i-1] < 0) {
        _nodes[k] += 1;
        std::cout << "Orbital " << k << ": Found node at i=" << alist_i[i] << ", r = " << (*_g)(alist_i[i]) << std::endl;
      }
    }
  }

  VectorXld ParN(E.size()+L.size());
  ParN.head(E.size()) = Fn;
  ParN.tail(L.size()) = Sn;

  VectorXld dPar(ParN.size());
  dPar.setZero();

  MatrixXld J(ParN.size(), ParN.size());
  J.setZero();

  VectorXld probe_dE(E.size());
  probe_dE.setZero();
  VectorXld probe_dLambda(L.size());
  probe_dLambda.setZero();

  int iterE = _historyE.size()-1;
  for (int k = 0; k < ParN.size(); ++k) {
    VectorXld EdE = E;
    VectorXld LdL = L;
  
    if (k < E.size()) {
      probe_dE(k) = E(k)*1e-2/((ldouble) _o[k]->n());
      if (iterE >= 1) {
        if (_historyE[iterE-1](k) != 0 && _historyE[iterE](k) != 0)
          probe_dE(k) = 0.1*(_historyE[iterE-1](k) - _historyE[iterE](k));
      }

      EdE(k) += probe_dE(k);
    } else {
      probe_dLambda(k - E.size()) = 1e-2;
      LdL(k - E.size()) += probe_dLambda(k - E.size());
    }

    VectorXld Fd;
    VectorXld Sd(L.size());
    ldouble sumFd2 = solveStandard(EdE, LdL, Sd, Fd, matchedSt);

    // two criteria to satisfy:
    // F = 0 and S = 0
    if (k < E.size()) {
      for (int j = 0; j < ParN.size(); ++j) {
        if (j < E.size())
          J(j, k) = (Fd(j) - Fn(j))/probe_dE(k);
        else
          J(j, k) = (Sd(j - _o.size()) - Sn(j - _o.size()))/probe_dE(k);
      }
    } else {
      for (int j = 0; j < ParN.size(); ++j) {
        if (j < E.size())
          J(j, k) = (Fd(j) - Fn(j))/probe_dLambda(k - E.size());
        else
          J(j, k) = (Sd(j - E.size()) - Sn(j - E.size()))/probe_dLambda(k - E.size());
      }
    }
  }

  dPar = J.fullPivLu().solve(ParN);

  ldouble alpha = -1;
  for (int k = 0; k < L.size(); ++k) {
    _dlambda[k] = alpha*gamma*dPar(_o.size()+k);
    if (std::fabs(_dlambda[k]) > 0.5) _dlambda[k] = 0.5*_dlambda[k]/std::fabs(_dlambda[k]);
    std::cout << "INFO: Lagrange multiplier " << k << " (with the Newton-Raphson method), dlambda = " << _dlambda[k] << " (probe dlambda = " << probe_dLambda(k) << ")" << std::endl;
  }
  for (int k = 0; k < E.size(); ++k) {
    _dE[k] = alpha*gamma*dPar(k);
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "INFO: Orbital " << k << " (with the Newton-Raphson method), dE = " << _dE[k] << " (probe dE = " << probe_dE(k) << ")" << std::endl;
  }

  VectorXld tmp = E;
  for (int k = 0; k < _o.size(); ++k) {

    int idx = _om.index(k);

    int nodes_found = _nodes[k];
    int nodes_target = _o[k]->n() - _o[k]->l() - 1;

    ldouble orb_n = _o[k]->n();
    ldouble orb_n_eff = _nodes[k] + _o[k]->l() + 1;
    
    ldouble &orb_dE = _dE[k];
    ldouble orb_E = _o[k]->E();

    if (std::abs(nodes_found - nodes_target) == 1 && std::fabs(orb_dE/orb_E) > 0.02) { // only one node off
      std::cout << "INFO: One node off in orbital " << k << "." << std::endl;
      ldouble EE = orb_E + orb_dE; // normal shift
      if (EE < _Emax[k] && EE > _Emin[k]) {
        orb_dE = EE - orb_E;
      } else {
        // try E (n'/n)^2.5
        EE = orb_E*std::pow( (orb_n_eff/orb_n), 2.5);
        if (EE < _Emax[k] && EE > _Emin[k]) {
          orb_dE = EE - orb_E;
        } else {
          // try +/ delta E / 2^k for several k until one value is in range
          for (double kk = 0; kk <= 20; kk++) {
            EE = orb_E + orb_dE*std::pow(2, -kk);
            if (EE < _Emax[k] && EE > _Emin[k]) {
              orb_dE = EE - orb_E;
              break;
            }
          }
        }
      }
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else if ((nodes_found - nodes_target) == 0) { // correct number of nodes
      // try +/ delta E / 2^k for several k until one value is in range
      for (double kk = 0; kk <= 20; kk++) {
        ldouble EE = orb_E + orb_dE*std::pow(2, -kk);
        if (EE < _Emax[k] && EE > _Emin[k]) {
          orb_dE = EE - orb_E;
          break;
        }
      }
    } else if (nodes_found < nodes_target) { // too few nodes
      std::cout << "INFO: Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;

      ldouble delta = 1 - orb_E/_Emin[k];
      _Emin[k] = orb_E;

      _Emin_n[k] = orb_E*std::pow(orb_n_eff/orb_n, 2.5);
      if (delta < 0.05) _Emin_n[k] = _Emin[k]*std::pow(orb_n_eff/orb_n, 2.5);

      if (_Emin_n[k] < _Emax[k] && _Emin_n[k] > _Emin[k]) {
        orb_dE = _Emin_n[k] - orb_E;
      } else {
        orb_dE = -orb_E + (_Emin[k] + _Emax[k])*0.5;
      }

      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else if (nodes_found > nodes_target) {
      std::cout << "INFO: Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      ldouble delta = 1 - orb_E/_Emax[k];
      _Emax[k] = orb_E;

      _Emax_n[k] = orb_E*std::pow(orb_n_eff/orb_n, 2.5);
      if (delta < 0.05) _Emax_n[k] = _Emax[k]*std::pow(orb_n_eff/orb_n, 2.5);

      if (_Emin_n[k] < _Emax[k] && _Emin_n[k] > _Emin[k]) {
        orb_dE = _Emin_n[k] - orb_E;
      } else {
        orb_dE = -orb_E + (_Emin[k] + _Emax[k])*0.5;
      }

      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    }
  }
  _historyE.push_back(tmp);
  _historyF.push_back(Fn);
  _historyL.push_back(L);
  
  return Fn.squaredNorm()+Sn.squaredNorm();
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepStandardMinim(ldouble gamma) {
  int N = _om.N();

  VectorXld E(_o.size());
  for (int k = 0; k < _o.size(); ++k) E(k) = _o[k]->E();
  VectorXld L(_lambda.size());
  for (int k = 0; k < _lambda.size(); ++k) L(k) = _lambda[k];

  VectorXld Fn(E.size());
  VectorXld Sn(L.size());
  ldouble sumFn = solveStandard(E, L, Sn, Fn, matchedSt);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);

    ldouble a_max = 0;
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matchedSt[idx][i];
      if (std::fabs((*_o[k])(i)) > a_max) {
        a_max = std::fabs((*_o[k])(i));
      }
    }
    std::vector<ldouble> alist;
    std::vector<ldouble> alist_i;
    for (int i = 0; i < _g->N(); ++i) {
      if (std::fabs((*_o[k])(i)) < 0.01*a_max) continue;
      alist.push_back((*_o[k])(i));
      alist_i.push_back(i);
    }
    for (int i = 0; i < alist.size(); ++i) {
      if (i >= 1 && alist[i]*alist[i-1] < 0) {
        _nodes[k] += 1;
        std::cout << "Orbital " << k << ": Found node at i=" << alist_i[i] << ", r = " << (*_g)(alist_i[i]) << std::endl;
      }
    }
  }

  VectorXld ParN(E.size()+L.size());
  ParN.head(E.size()) = Fn;
  ParN.tail(L.size()) = Sn;

  VectorXld dPar(ParN.size());
  dPar.setZero();

  VectorXld gradP(ParN.size());
  gradP.setZero();

  MatrixXld Hn(ParN.size(), ParN.size());
  Hn.setZero();

  VectorXld probe_dE(E.size());
  probe_dE.setZero();
  VectorXld probe_dLambda(L.size());
  probe_dLambda.setZero();

  int iterE = _historyE.size()-1;
  for (int k = 0; k < ParN.size(); ++k) {
    VectorXld EdE = E;
    VectorXld LdL = L;
  
    if (k < E.size()) {
      probe_dE(k) = E(k)*1e-2/((ldouble) _o[k]->n());
      if (iterE >= 1) {
        if (_historyE[iterE-1](k) != 0 && _historyE[iterE](k) != 0)
          probe_dE(k) = 0.1*(_historyE[iterE-1](k) - _historyE[iterE](k));
      }
    } else {
      probe_dLambda(k - E.size()) = 1e-2;
    }
  }

  // nominal
  //VectorXld EdE1 = E;
  //VectorXld LdL1 = L;
  //VectorXld Fn1(E.size());
  //VectorXld Sn1(L.size());
  //ldouble sumFn1 = solveStandard(EdE1, LdL1, Sn1, Fn1, matchedSt);
  // calculate gradient
  for (int k = 0; k < ParN.size(); ++k) {
    VectorXld EdE2 = E;
    VectorXld LdL2 = L;
    if (k < E.size()) {
      EdE2(k) += probe_dE(k);
    } else {
      LdL2(k - E.size()) += probe_dLambda(k - E.size());
    }

    VectorXld Fd2;
    VectorXld Sd2(L.size());
    ldouble sumFn2 = solveStandard(EdE2, LdL2, Sd2, Fd2, matchedSt);

    if (k < E.size()) {
      gradP(k) = (sumFn2 - sumFn)/probe_dE(k);
    } else {
      gradP(k) = (sumFn2 - sumFn)/probe_dLambda(k - E.size());
    }

    //if (_reestimateHessian) {
      VectorXld gradP2(ParN.size());
      gradP2.setZero();
      for (int j = 0; j < ParN.size(); ++j) {
        VectorXld EdE1 = E;
        VectorXld LdL1 = L;
        if (j < E.size()) {
          EdE1(j) += probe_dE(j);
        } else {
          LdL1(j - E.size()) += probe_dLambda(j - E.size());
        }
        VectorXld Fd1;
        VectorXld Sd1(L.size());
        ldouble sumFd1 = solveStandard(EdE1, LdL1, Sd1, Fd1, matchedSt);

        VectorXld EdE2 = EdE1;
        VectorXld LdL2 = LdL1;
        if (k < E.size()) {
          EdE2(k) += probe_dE(k);
        } else {
          LdL2(k - E.size()) += probe_dLambda(k - E.size());
        }
        VectorXld Fd2;
        VectorXld Sd2(L.size());
        ldouble sumFd2 = solveStandard(EdE2, LdL2, Sd2, Fd2, matchedSt);

        if (k < E.size()) {
          gradP2(k) = (sumFd2 - sumFd1)/probe_dE(k);
        } else {
          gradP2(k) = (sumFd2 - sumFd1)/probe_dLambda(k - E.size());
        }

        if (j < E.size()) {
          Hn(k, j) = (gradP2(k) - gradP(k))/probe_dE(j);
        } else {
          Hn(k, j) = (gradP2(k) - gradP(k))/probe_dLambda(j - E.size());
        }
      }
    //} else { // update Hessian
    //  VectorXld curr_gradP = gradP;
    //  VectorXld prev_gradP = _gradP;
    //  VectorXld dg = curr_gradP - prev_gradP;
    //  VectorXld dx = ParN - _ParN;
    //  Hn = _H + dg*dg.transpose()/(dg.transpose()*dx) - _H*dx*dx.transpose()*_H.transpose()/(dx.transpose()*_H*dx);
    //}
  }

  dPar = -Hn.fullPivLu().solve(gradP);
  ldouble alpha = 1;

  //int alphaIter = 0;
  //while (alphaIter++ < 10) {
  //  ldouble alphaStep = 0.01;
  //  VectorXld EdE = E;
  //  VectorXld LdL = L;
  //  for (int k = 0; k < ParN.size(); ++k) {
  //    if (k < E.size()) {
  //      EdE(k) += alpha*dPar(k);
  //    } else { 
  //      LdL(k - E.size()) += alpha*dPar(k);
  //    }
  //  }
  //  // solve it again
  //  VectorXld Fna(E.size());
  //  VectorXld Sna(L.size());
  //  ldouble sumFna = solveStandard(EdE, LdL, Sna, Fna, matchedSt);
  //  // step alpha * dPar
  //  for (int k = 0; k < ParN.size(); ++k) {
  //    if (k < E.size()) {
  //      EdE(k) += alphaStep*dPar(k);
  //    } else { 
  //      LdL(k - E.size()) += alphaStep*dPar(k);
  //    }
  //  }
  //  VectorXld Fnb(E.size());
  //  VectorXld Snb(L.size());
  //  ldouble sumFnb = solveStandard(EdE, LdL, Snb, Fnb, matchedSt);
  //  ldouble gradAlpha = 0;
  //  for (int k = 0; k < ParN.size(); ++k) {
  //    if (k < E.size()) {
  //      gradAlpha += 2 * Fna(k) * (Fnb(k) - Fna(k))/alphaStep;
  //    } else {
  //      gradAlpha += 2 * Sna(k - E.size()) * (Snb(k - E.size()) - Sna(k - E.size()))/alphaStep;
  //    }
  //  }
  //  // get gradient
  //  //ldouble gradAlpha = (sumFnb - sumFna)/alphaStep;
  //  if (gradAlpha == 0 || !(gradAlpha == gradAlpha)) break;
  //  alpha = alpha - sumFna/gradAlpha;
  //  std::cout << "Updated alpha to " << alpha << std::endl;
  //}

  for (int k = 0; k < L.size(); ++k) {
    _dlambda[k] = alpha*gamma*dPar(_o.size()+k);
    if (std::fabs(_dlambda[k]) > 0.5) _dlambda[k] = 0.5*_dlambda[k]/std::fabs(_dlambda[k]);
    std::cout << "INFO: Lagrange multiplier " << k << " (with the Newton-Raphson method), dlambda = " << _dlambda[k] << " (probe dlambda = " << probe_dLambda(k) << ")" << std::endl;
  }
  for (int k = 0; k < E.size(); ++k) {
    _dE[k] = alpha*gamma*dPar(k);
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "INFO: Orbital " << k << " (with the Newton-Raphson method), dE = " << _dE[k] << " (probe dE = " << probe_dE(k) << ")" << std::endl;
  }
  _H = Hn;
  _gradP = gradP;
  _ParN = ParN;
  _reestimateHessian = false;

  VectorXld tmp = E;
  for (int k = 0; k < _o.size(); ++k) {

    int idx = _om.index(k);

    int nodes_found = _nodes[k];
    int nodes_target = _o[k]->n() - _o[k]->l() - 1;

    ldouble orb_n = _o[k]->n();
    ldouble orb_n_eff = _nodes[k] + _o[k]->l() + 1;
    
    ldouble &orb_dE = _dE[k];
    ldouble orb_E = _o[k]->E();

    if (std::abs(nodes_found - nodes_target) == 1 && std::fabs(orb_dE/orb_E) > 0.02) { // only one node off
      std::cout << "INFO: One node off in orbital " << k << "." << std::endl;
      ldouble EE = orb_E + orb_dE; // normal shift
      if (EE < _Emax[k] && EE > _Emin[k]) {
        orb_dE = EE - orb_E;
      } else {
        // try E (n'/n)^2.5
        EE = orb_E*std::pow( (orb_n_eff/orb_n), 2.5);
        if (EE < _Emax[k] && EE > _Emin[k]) {
          orb_dE = EE - orb_E;
        } else {
          // try +/ delta E / 2^k for several k until one value is in range
          for (double kk = 0; kk <= 20; kk++) {
            EE = orb_E + orb_dE*std::pow(2, -kk);
            if (EE < _Emax[k] && EE > _Emin[k]) {
              orb_dE = EE - orb_E;
              break;
            }
          }
        }
      }
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else if ((nodes_found - nodes_target) == 0) { // correct number of nodes
      // try +/ delta E / 2^k for several k until one value is in range
      for (double kk = 0; kk <= 20; kk++) {
        ldouble EE = orb_E + orb_dE*std::pow(2, -kk);
        if (EE < _Emax[k] && EE > _Emin[k]) {
          orb_dE = EE - orb_E;
          break;
        }
      }
    } else if (nodes_found < nodes_target) { // too few nodes
      std::cout << "INFO: Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;

      ldouble delta = 1 - orb_E/_Emin[k];
      _Emin[k] = orb_E;

      _Emin_n[k] = orb_E*std::pow(orb_n_eff/orb_n, 2.5);
      if (delta < 0.05) _Emin_n[k] = _Emin[k]*std::pow(orb_n_eff/orb_n, 2.5);

      if (_Emin_n[k] < _Emax[k] && _Emin_n[k] > _Emin[k]) {
        orb_dE = _Emin_n[k] - orb_E;
      } else {
        orb_dE = -orb_E + (_Emin[k] + _Emax[k])*0.5;
      }

      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else if (nodes_found > nodes_target) {
      std::cout << "INFO: Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      ldouble delta = 1 - orb_E/_Emax[k];
      _Emax[k] = orb_E;

      _Emax_n[k] = orb_E*std::pow(orb_n_eff/orb_n, 2.5);
      if (delta < 0.05) _Emax_n[k] = _Emax[k]*std::pow(orb_n_eff/orb_n, 2.5);

      if (_Emin_n[k] < _Emax[k] && _Emin_n[k] > _Emin[k]) {
        orb_dE = _Emin_n[k] - orb_E;
      } else {
        orb_dE = -orb_E + (_Emin[k] + _Emax[k])*0.5;
      }

      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    }
  }
  _historyE.push_back(tmp);
  _historyF.push_back(Fn);
  _historyL.push_back(L);
  
  return Fn.squaredNorm()+Sn.squaredNorm();
}

