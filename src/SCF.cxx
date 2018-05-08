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

#include <Python.h>
using namespace boost;

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <cstdlib>

SCF::SCF()
  : _g(new Grid(expGrid, 1e-1, 10, 1e-3)), _Z(1), _om(*_g, _o), _lsb(*_g, _o, icl, _om), _irs(*_g, _o, icl, _om), _iss(*_g, _o, icl, _om), _igs(*_g, _o, icl, _om) {
  _own_grid = true;
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
  _gamma_scf = 0.2;
  _method = 3;
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
  if (m > 3) m = 3;
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
    strMethod = "Iterative Numerov with Gordon method for initial condition (http://aip.scitation.org/doi/pdf/10.1063/1.436421)";
  } else if (_method == 2) {
    strMethod = "Iterative Renormalised Numerov (http://aip.scitation.org/doi/pdf/10.1063/1.436421)";
  } else if (_method == 3) {
    strMethod = "Iterative Standard Numerov with non-homogeneous term";
  }

  _historyE.clear();
  _historyF.clear();

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
    } else if (_method == 3) {
      gamma = 0.5*(1 - std::exp(-(nStep+1)/5.0));
      F = stepStandard(gamma);
    }

    // change orbital energies
    std::cout << "Orbital energies at step " << nStep << ", with constraint = " << std::setw(16) << F << ", method = " << strMethod << "." << std::endl;
    std::cout << std::setw(5) << "Index" << " " << std::setw(16) << "Energy (H)" << " " << std::setw(16) << "next energy (H)" << " " << std::setw(16) << "Min. (H)" << " " << std::setw(16) << "Max. (H)" << " " << std::setw(5) << "nodes" << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      ldouble stepdE = _dE[k];
      ldouble newE = (_o[k]->E()+stepdE);
      std::cout << std::setw(5) << k << " " << std::setw(16) << std::setprecision(12) << _o[k]->E() << " " << std::setw(16) << std::setprecision(12) << newE << " " << std::setw(16) << std::setprecision(12) << _Emin[k] << " " << std::setw(16) << std::setprecision(12) << _Emax[k] << " " << std::setw(5) << _nodes[k] << std::endl;
      _o[k]->E(newE);
    }

    if (std::fabs(*std::max_element(_dE.begin(), _dE.end(), [](ldouble a, ldouble b) -> bool { return std::fabs(a) < std::fabs(b); } )) < F0stop) break;
    //if (std::fabs(F) < F0stop) break;
  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepGordon(ldouble gamma) {
  int N = _o.size();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = -1e-3;
    E[k] = _o[k]->E();
    l[k] = _o[k]->l();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, E);

  ldouble Fn = _igs.solve(E, l, Fmn, Kmn, matched);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matched[i](idx);
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
    //std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //std::fabs(_Z*_Z*0.5/std::pow(_nodes[k], 2) - _Z*_Z*0.5/std::pow(_nodes[k]+1, 2));
      std::cout << "Orbital " << k << ", new dE = " << _dE[k] << std::endl;
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //-std::fabs(_Z*_Z*0.5/std::pow(_nodes[k], 2) - _Z*_Z*0.5/std::pow(_nodes[k]+1, 2));
      std::cout << "Orbital " << k << ", new dE = " << _dE[k] << std::endl;
    } else {
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }
    }
  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepRenormalised(ldouble gamma) {
  int N = _om.N();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  for (int k = 0; k < _o.size(); ++k) {
    E[k] = _o[k]->E();
    l[k] = _o[k]->l();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<VectorXld> matched;
  std::vector<int> Rnodes;
  calculateFMatrix(Fmn, Kmn, E);

  VectorXld Fn = _irs.solve(E, l, Fmn, Kmn, matched, Rnodes);
  std::cout << "INFO: Total nodes: " << Rnodes[0] << std::endl;

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matched[i](idx);
      if (i >= 10 && i < _g->N() - 4 && matched[i](idx)*matched[i-1](idx) <= 0) {
        _nodes[k] += 1;
        std::cout << "Orbital " << k << ": Found node at i=" << i << ", r = " << (*_g)(i) << std::endl;
      }
    }
  }

  int iterE = _historyE.size()-1;
  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    _dE[k] = 0;
    dE[k] = _o[k]->E()*1e-2/((ldouble) _o[k]->n());
    if (iterE >= 1) { // take a test E variation from the previous tested points
      dE[k] = _historyE[iterE-1][k] - _historyE[iterE][k];
      if (dE[k] == 0)
        dE[k] = _o[k]->E()*1e-2/((ldouble) _o[k]->n());
    }
  }
  VectorXld dEv(_o.size());
  for (int k = 0; k < _o.size(); ++k) {
    std::vector<ldouble> EdE = E;
    EdE[k] += dE[k];
  
    std::vector<MatrixXld> Fmd;
    std::vector<MatrixXld> Kmd;
    std::vector<int> Rnodesd;
    calculateFMatrix(Fmd, Kmd, EdE);

    // recalculate the function being minimized, because we are changing energy for orbital k independently from the others
    VectorXld Fd = _irs.solve(EdE, l, Fmd, Kmd, matched, Rnodesd);
    dEv(k) = Fn(k)*dE[k]/(Fd(k) - Fn(k));
  }

  // check node count
  for (int k = 0; k < _o.size(); ++k) {
    int idx = _om.index(k);
    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E()+1e-15;
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      std::cout << "Orbital " << k << ", new dE = " << _dE[k] << std::endl;
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E()-1e-15;
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      std::cout << "Orbital " << k << ", new dE = " << _dE[k] << std::endl;
    } else {
  
      _dE[k] = -gamma*dEv(k);

      std::cout << "Orbital " << k << " (with secant method), dE = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }
    }
  }

  VectorXld tmp;
  tmp.resize(_o.size());
  for (int k = 0; k < _o.size(); ++k) {
    tmp(k) = _o[k]->E();
  }
  _historyE.push_back(tmp);
  _historyF.push_back(Fn);

  return Fn.norm();
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepSparse(ldouble gamma) {
  // 1) build sparse matrix _A
  // 2) build sparse matrix _b
  if (_isSpinDependent) {
    _lsb.prepareMatrices(_A, _b0, _pot, _vsum_up, _vsum_dw);
  } else {
    _lsb.prepareMatrices(_A, _b0, _pot, _vd, _vex);
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
    for (int i = 0; i < _g->N(); ++i) {
      if (i >= 10  && i < _g->N() - 4 && (*_o[k])(i)*(*_o[k])(i-1) <= 0) {
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

  VectorXld Fn;
  if (_isSpinDependent) {
    Fn = _iss.solve(E, _pot, _vsum_up, _vsum_dw, matchedSt);
  } else {
    Fn = _iss.solve(E, _pot, _vd, _vex, matchedSt);
  }

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matchedSt[idx][i];
      if (i >= 10 && i < _g->N() - 4 && (*_o[k])(i)*(*_o[k])(i-1) < 0) {
        _nodes[k] += 1;
        std::cout << "Orbital " << k << ": Found node at i=" << i << ", r = " << (*_g)(i) << std::endl;
      }
    }
  }

  /*
  // to get the next energy, based on the procedure described in page 65 of:
  // L.V. CHERNYSHEVA and N.A. CHEREPKOV and N.A. CHEREPKOV, Computer Physics Communications 11(1976) 57—73
  // SELF-CONSISTENT FIELD HARTREE—FOCK PROGRAM FOR ATOMS
  int iterE = _historyE.size()-1;
  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    int idx = _om.index(k);

    bool foundSolution = false;
    if (iterE >= 2) { // if there are at least 3 points, fit a parabola

      VectorXld hE, hF;
      hE.resize(3);
      hF.resize(3);

      // could use those, but as other orbitals change, the correlation can cause instabilities
      hE(0) = _historyE[iterE-0](idx);     hF(0) = _historyF[iterE-0](idx);
      hE(1) = _historyE[iterE-1](idx);     hF(1) = _historyF[iterE-1](idx);
      hE(2) = _historyE[iterE-2](idx);     hF(2) = _historyF[iterE-2](idx);

      // it is better to recalculate previous steps, because the other orbitals energy changed
      std::vector<ldouble> EdE = E;
      VectorXld Fd;

      EdE = E;
      EdE[k] += - hE(0) + hE(1);
      if (_isSpinDependent) {
        Fd = _iss.solve(EdE, _pot, _vsum_up, _vsum_dw, matchedSt);
      } else {
        Fd = _iss.solve(EdE, _pot, _vd, _vex, matchedSt);
      }
      hE(1) = EdE[k];    hF(1) = Fd(idx);

      EdE = E;
      EdE[k] += - hE(0) + hE(2);
      if (_isSpinDependent) {
        Fd = _iss.solve(EdE, _pot, _vsum_up, _vsum_dw, matchedSt);
      } else {
        Fd = _iss.solve(EdE, _pot, _vd, _vex, matchedSt);
      }
      hE(2) = EdE[k];    hF(2) = Fd(idx);

      MatrixXld A;
      VectorXld F;
      A.resize(3, 3);
      F.resize(3);

      // f(E) = a E^2 + b E + c
      // e = sum_k=0^3 (f(E_k) - F_k)^2
      // grad e = 0
      // de/da = 2 sum_k (f(E_k) - F_k) (E_k^2) = 2 a sum_k E_k^4 + 2 b sum_k E_k^3 + 2 c sum_k E_k^2 - sum_k F_k E_k^2
      // de/db = 2 sum_k (f(E_k) - F_k) (E_k)   = 2 a sum_k E_k^3 + 2 b sum_k E_k^2 + 2 c sum_k E_k   - sum_k F_k E_k
      // de/dc = 2 sum_k (f(E_k) - F_k) (1)     = 2 a sum_k E_k^2 + 2 b sum_k E_k   + 2 c sum_k 1     - sum_k F_k
      A(0,0) = 2*(std::pow(hE(0), 4) + std::pow(hE(1), 4) + std::pow(hE(2), 4));
      A(0,1) = 2*(std::pow(hE(0), 3) + std::pow(hE(1), 3) + std::pow(hE(2), 3));
      A(0,2) = 2*(std::pow(hE(0), 2) + std::pow(hE(1), 2) + std::pow(hE(2), 2));
      F(0) = hF(0)*std::pow(hE(0), 2) + hF(1)*std::pow(hE(1), 2) + hF(2)*std::pow(hE(2), 2);
      A(1,0) = 2*(std::pow(hE(0), 3) + std::pow(hE(1), 3) + std::pow(hE(2), 3));
      A(1,1) = 2*(std::pow(hE(0), 2) + std::pow(hE(1), 2) + std::pow(hE(2), 2));
      A(1,2) = 2*(std::pow(hE(0), 1) + std::pow(hE(1), 1) + std::pow(hE(2), 1));
      F(1) = hF(0)*std::pow(hE(0), 1) + hF(1)*std::pow(hE(1), 1) + hF(2)*std::pow(hE(2), 1);
      A(2,0) = 2*(std::pow(hE(0), 2) + std::pow(hE(1), 2) + std::pow(hE(2), 1));
      A(2,1) = 2*(std::pow(hE(0), 1) + std::pow(hE(1), 1) + std::pow(hE(2), 1));
      A(2,2) = 2*(std::pow(hE(0), 0) + std::pow(hE(1), 0) + std::pow(hE(2), 0));
      F(2) = hF(0)*std::pow(hE(0), 0) + hF(1)*std::pow(hE(1), 0) + hF(2)*std::pow(hE(2), 0);

      ldouble a, b, c;
      if (std::fabs(A.determinant()) > 1e-14) {
        VectorXld p = A.inverse()*F;
        a = p(0);
        b = p(1);
        c = p(2);
        ldouble delta = b*b - 4*a*c;
        if (delta > 0 && std::fabs(a) > 1e-14 && !std::isnan(a) && !std::isnan(b) && !std::isnan(c)) {
          ldouble E1 = (-b + std::sqrt(delta))/(2*a);
          ldouble E2 = (-b - std::sqrt(delta))/(2*a);
          if (E1 > 0 && E2 < 0) {
            _dE[k] = E2 - _o[k]->E();
            if (std::fabs(_dE[k]) < 0.5)
              foundSolution = true;
          } else if (E1 < 0 && E2 > 0) {
            _dE[k] = E1 - _o[k]->E();
            if (std::fabs(_dE[k]) < 0.5)
              foundSolution = true;
          } else if (E1 < 0 && E2 < 0) {
            ldouble dE1 = E1 - _o[k]->E();
            ldouble dE2 = E2 - _o[k]->E();
            if (std::fabs(dE1) < std::fabs(dE2)) {
              _dE[k] = dE1;
            } else {
              _dE[k] = dE2;
            }
            if (std::fabs(_dE[k]) < 0.5)
              foundSolution = true;
          }
          _dE[k] *= gamma;
        }
      }
      if (foundSolution)
        std::cout << "INFO: Orbital " << k << " (with parabola fit), dE = " << _dE[k] << " (det(A), a,b,c = " << A.determinant() << ", " << a << ", " << b << ", " << c << ")" << std::endl;
      else
        std::cout << "INFO: Orbital " << k << " (failed parabola fit), (det(A), a,b,c = " << A.determinant() << ", " << a << ", " << b << ", " << c << ")" << std::endl;
    }

    // try the secant method
    if (!foundSolution) {
      // if we have no past iterations or the differences are too small, calculate the effect
      // of shifting E by an arbitrary amount
      dE[k] = E[k]*1e-2/((ldouble) _o[k]->n());
      if (iterE >= 1) {
        dE[k] = _historyE[iterE-1][k] - _historyE[iterE][k];
      }
      if (dE[k] == 0)
        dE[k] = E[k]*1e-2/_o[k]->n();

      std::vector<ldouble> EdE = E;
      EdE[k] += dE[k];

      VectorXld Fd;
      if (_isSpinDependent) {
        Fd = _iss.solve(EdE, _pot, _vsum_up, _vsum_dw, matchedSt);
      } else {
        Fd = _iss.solve(EdE, _pot, _vd, _vex, matchedSt);
      }

      int nodes = 0;
      for (int i = 0; i < _g->N(); ++i) {
        if (i >= 10 && i < _g->N() - 4 && (*_o[k])(i)*(*_o[k])(i-1) < 0) {
          nodes += 1;
        }
      }

      _dE[k] = 0;
      if (nodes != _nodes[k]) {
        std::cout << "INFO: Found " << nodes << " nodes != " << _nodes[k] << ", after probe step in secant method for orbital " << k << "." << std::endl;
      } else {
        if (std::fabs(Fn(idx) - Fd(idx)) != 0) {
          _dE[k] = Fn(idx)*(-dE[k])/(Fn(idx) - Fd(idx));
        }
        _dE[k] *= -gamma;
      }

      if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
      std::cout << "INFO: Orbital " << k << " (with secant method), dE = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
    }

    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/((ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      _historyE.clear();
      _historyF.clear();
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/( (ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      _historyE.clear();
      _historyF.clear();
    } else {
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }

      VectorXld tmp;
      tmp.resize(_o.size());
      for (int k = 0; k < _o.size(); ++k) {
        tmp(k) = _o[k]->E();
      }
      _historyE.push_back(tmp);
      _historyF.push_back(Fn);
    }
  }
  */

  VectorXld dEv(_o.size());
  dEv.setZero();
  MatrixXld J(_o.size(), _o.size());
  J.setZero();

  int iterE = _historyE.size()-1;
  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    int idx = _om.index(k);
    dE[k] = E[k]*1e-2/((ldouble) _o[k]->n());
    if (iterE >= 1) {
      dE[k] = _historyE[iterE-1][k] - _historyE[iterE][k];
    }
    if (dE[k] == 0)
      dE[k] = E[k]*1e-2/_o[k]->n();

    std::vector<ldouble> EdE = E;
    EdE[k] += dE[k];

    VectorXld Fd;
    if (_isSpinDependent) {
      Fd = _iss.solve(EdE, _pot, _vsum_up, _vsum_dw, matchedSt);
    } else {
      Fd = _iss.solve(EdE, _pot, _vd, _vex, matchedSt);
    }
    for (int j = 0; j < _o.size(); ++j) {
      J(j, k) = (Fd(j) - Fn(j))/dE[k];
    }

  }
  dEv = J.inverse()*Fn;

  for (int k = 0; k < _o.size(); ++k) {
    _dE[k] = -gamma*dEv(k);
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "INFO: Orbital " << k << " (with the Newton-Raphson method), dE = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
  }

  for (int k = 0; k < _o.size(); ++k) {
    int idx = _om.index(k);
    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/((ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      _historyE.clear();
      _historyF.clear();
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/( (ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      _historyE.clear();
      _historyF.clear();
    } else {
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }

      VectorXld tmp;
      tmp.resize(_o.size());
      for (int k = 0; k < _o.size(); ++k) {
        tmp(k) = _o[k]->E();
      }
      _historyE.push_back(tmp);
      _historyF.push_back(Fn);
    }
  }
  
  ldouble F = Fn.norm();
  return F;
}

