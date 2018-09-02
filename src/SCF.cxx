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
using namespace boost;

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <cstdlib>

#include "HFException.h"

SCF::SCF(ldouble Z)
  : _g(new Grid(expGrid, 1.0/32.0, (int) ((std::log(30.0) + 6 + std::log(Z))/(1.0/32.0))+1, std::exp(-6)/Z)),
  _Z(Z), _om(*_g, _o), _lsb(*_g, _o, icl, _om), _iss(*_g, _o, icl, _om) {
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
  while (nStep < Niter) {
    gamma = 0.5*(1 - std::exp(-(nStep+1)/20.0));
    // compute sum of squares of F(x_old)
    nStep += 1;
    if (_method == 0) {
      F = stepSparse(gamma);
    } else if (_method == 1) {
      gamma = 0.5*(1 - std::exp(-(nStep+1)/5.0));
      F = stepStandard(gamma);
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
  
    bool stop = std::fabs(*std::max_element(_dE.begin(), _dE.end(), [](ldouble a, ldouble b) -> bool { return std::fabs(a) < std::fabs(b); } )) < F0stop;
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
      if (std::fabs((*_o[k])(i)) < 0.05*a_max) continue;
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

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepStandard(ldouble gamma) {
  int N = _om.N();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  for (int k = 0; k < _o.size(); ++k) {
    E[k] = _o[k]->E();
    l[k] = _o[k]->l();
  }

  VectorXld Sn(_lambda.size());
  Sn.setZero();

  VectorXld Fn;
  if (_isSpinDependent) {
    Fn = _iss.solve(E, _pot, _vsum_up, _vsum_dw, _lambda, _lambdaMap, matchedSt);
  } else {
    Fn = _iss.solve(E, _pot, _vd, _vex, _lambda, _lambdaMap, matchedSt);
  }

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (k1 <= k2) continue;
      if (_o[k1]->l() != _o[k2]->l()) continue;
      int lidx = _lambdaMap[k1*100+k2];
      for (int ir = 0; ir < _g->N()-1; ++ir) {
        if (_g->isLog()) // y = psi * sqrt(r) and dr = r dx, so psi1 * psi2 * r^2 * dr = y1 * y2 / r * r^2 * r * dx = y1 y2 r^2 dx
          Sn(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2)*_g->dx();
        else if (_g->isLin()) // y = psi and dr = dx, so psi1 psi2 r^2 dr = y1 y2 r^2 dx
          Sn(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2)*_g->dx();
      }
    }
  }
  //std::cout << "First derivative constraint" << std::endl;
  //std::cout << Fn << std::endl;
  //std::cout << "Overlap matrix off-diagonal elements" << std::endl;
  //std::cout << Sn << std::endl;

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
      if (std::fabs((*_o[k])(i)) < 0.05*a_max) continue;
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

  VectorXld ParN(_o.size()+_lambda.size());
  ParN.setZero();
  for (int k = 0; k < _o.size(); ++k) {
    ParN(k) = Fn(k);
  }
  for (int k = 0; k < _lambda.size(); ++k) {
    ParN(_o.size()+k) = Sn(k);
  }
  VectorXld dPar(_o.size()+_lambda.size());
  dPar.setZero();
  MatrixXld J(_o.size()+_lambda.size(), _o.size()+_lambda.size());
  J.setZero();

  int iterE = _historyE.size()-1;
  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size()+_lambda.size(); ++k) {
    std::vector<ldouble> EdE = E;
    std::vector<ldouble> lambdad = _lambda;
  
    if (k < _o.size()) {
      dE[k] = 0;
      if (iterE >= 1) {
        if (_historyE[iterE-1](k) != 0 && _historyE[iterE](k) != 0)
          dE[k] = _historyE[iterE-1](k) - _historyE[iterE](k);
      }
      if (dE[k] == 0)
        dE[k] = E[k]*1e-2/((ldouble) _o[k]->n());

      EdE[k] += dE[k];
    } else {
      lambdad[k-_o.size()] += 1e-2;
    }

    VectorXld Fd;
    if (_isSpinDependent) {
      Fd = _iss.solve(EdE, _pot, _vsum_up, _vsum_dw, lambdad, _lambdaMap, matchedSt);
    } else {
      Fd = _iss.solve(EdE, _pot, _vd, _vex, lambdad, _lambdaMap, matchedSt);
    }

    VectorXld Sd(_lambda.size());
    Sd.setZero();
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      for (int k2 = 0; k2 < _o.size(); ++k2) {
        if (k1 <= k2) continue;
        if (_o[k1]->l() != _o[k2]->l()) continue;
        int lidx = _lambdaMap[k1*100+k2];
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          if (_g->isLog()) // y = psi * sqrt(r) and dr = r dx, so psi1 * psi2 * r^2 * dr = y1 * y2 / r * r^2 * r * dx = y1 y2 r^2 dx
            Sd(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2)*_g->dx();
          else if (_g->isLin()) // y = psi and dr = dx, so psi1 psi2 r^2 dr = y1 y2 r^2 dx
            Sd(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2)*_g->dx();
        }
      }
    }
    // two criteria to satisfy:
    // F = 0 and S = 0
    if (k < _o.size()) {
      for (int j = 0; j < _o.size() + _lambda.size(); ++j) {
        if (j < _o.size())
          J(j, k) = (Fd(j) - Fn(j))/dE[k];
        else
          J(j, k) = (Sd(j - _o.size()) - Sn(j - _o.size()))/dE[k];
      }
    } else {
      for (int j = 0; j < _o.size() + _lambda.size(); ++j) {
        if (j < _o.size())
          J(j, k) = (Fd(j) - Fn(j))/1e-2;
        else
          J(j, k) = (Sd(j - _o.size()) - Sn(j - _o.size()))/1e-2;
      }
    }
  }
  //std::cout << "Jacobian" << std::endl;
  //std::cout << J << std::endl;
  //std::cout << "Nominal minimisation function value" << std::endl;
  //std::cout << ParN << std::endl;
  //std::cout << "Jacobian inverse" << std::endl;
  //std::cout << J.inverse() << std::endl;
  //std::cout << "Jacobian singular values" << std::endl;
  //std::cout << J.jacobiSvd(ComputeThinU | ComputeThinV).singularValues() << std::endl;
  //std::cout << "Jacobian U and V" << std::endl;
  //std::cout << J.jacobiSvd(ComputeThinU | ComputeThinV).matrixU() << std::endl << J.jacobiSvd(ComputeThinU | ComputeThinV).matrixV() << std::endl;
  //dPar = J.inverse()*ParN;
  dPar = J.jacobiSvd(ComputeThinU | ComputeThinV).solve(ParN);
  //dPar = J.fullPivLu().solve(ParN);
  //std::cout << "Calculated step" << std::endl;
  //std::cout << dPar << std::endl;

  for (int k = 0; k < _lambda.size(); ++k) {
    _dlambda[k] = -gamma*dPar(_o.size()+k);
    std::cout << "INFO: Lagrange multiplier " << k << " (with the Newton-Raphson method), dlambda = " << _dlambda[k] << " (probe dlambda = " << 1e-2 << ")" << std::endl;
  }
  for (int k = 0; k < _o.size(); ++k) {
    _dE[k] = -gamma*dPar(k);
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "INFO: Orbital " << k << " (with the Newton-Raphson method), dE = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
  }

  VectorXld tmp;
  tmp.resize(_o.size());
  VectorXld tmpl;
  tmpl.resize(_lambda.size());

  for (int k = 0; k < _o.size(); ++k) {
    int idx = _om.index(k);
    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/((ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/( (ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      tmp(k) = 0;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else {
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }
  
      tmp(k) = _o[k]->E();
    }
  }
  for (int k = 0; k < _lambda.size(); ++k) {
    tmpl(k) = _lambda[k];
  }
  _historyE.push_back(tmp);
  _historyF.push_back(Fn);
  _historyL.push_back(tmpl);
  
  return Fn.squaredNorm()+Sn.squaredNorm();
}


