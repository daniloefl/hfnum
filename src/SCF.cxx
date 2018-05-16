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

  _historyL.clear();
  _historyE.clear();
  _historyF.clear();

  // create lambda map to flatten it
  // and resize it to be sure it has the proper size
  _lambdaMap.clear();
  int lcount = 0;
  if (!_isSpinDependent && (_method == 3 || _method == 2)) {
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

  _iss._i0.clear();
  _iss._i1.clear();

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
    std::cout << "Lagrange multipliers" << std::endl;
    std::cout << std::setw(10) << "Index" << " " << std::setw(16) << "Value" << " " << std::setw(16) << "Step" << std::endl;
    for (auto &k : _lambdaMap) {
      std::cout << std::setw(10) << k.first << " " << std::setw(16) << std::setprecision(12) << _lambda[k.second] << " " << std::setw(16) << std::setprecision(12) << _dlambda[k.second] << std::endl;
    }
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
  }

  if (_method == 3) {
    _iss._i0.resize(_o.size());
    _iss._i1.resize(_o.size());
    for (int k = 0; k < _o.size(); ++k) {
      _iss._i0[k] = _o[k]->getNorm(0, *_g);
      _iss._i1[k] = _o[k]->getNorm(1, *_g);
    }
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
  calculateFMatrix(Fmn, Kmn, E, _lambda);

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
    calculateFMatrix(Fmd, Kmd, EdE, _lambda);

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
  calculateFMatrix(Fmn, Kmn, E, _lambda);

  VectorXld Fn = _irs.solve(E, l, Fmn, Kmn, matched, Rnodes);
  std::cout << "INFO: Total nodes: " << Rnodes[0] << std::endl;

  VectorXld Sn(_lambda.size());
  Sn.setZero();

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      if (k1 <= k2) continue;
      if (_o[k1]->l() != _o[k2]->l()) continue;
      int lidx = _lambdaMap[k1*100+k2];
      for (int ir = 0; ir < _g->N()-1; ++ir) {
        ldouble dr = (*_g)(ir+1) - (*_g)(ir);
        if (_g->isLog())
          Sn(lidx) += matched[ir](k1)*matched[ir](k2)*std::pow((*_g)(ir), 2-1)*dr;
        else
          Sn(lidx) += matched[ir](k1)*matched[ir](k2)*std::pow((*_g)(ir), 2)*dr;
      }
    }
  }

  VectorXld dPar(_o.size()+_lambda.size());
  dPar.setZero();

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
  for (int k = 0; k < _o.size()+_lambda.size(); ++k) {
    std::vector<ldouble> EdE = E;
    std::vector<ldouble> lambdad = _lambda;
  
    if (k < _o.size()) {
      EdE[k] += dE[k];
    } else {
      lambdad[k-_o.size()] += 1e-2;
    }

    std::vector<MatrixXld> Fmd;
    std::vector<MatrixXld> Kmd;
    std::vector<int> Rnodesd;
    calculateFMatrix(Fmd, Kmd, EdE, lambdad);

    VectorXld Sd(_lambda.size());
    Sd.setZero();
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      for (int k2 = 0; k2 < _o.size(); ++k2) {
        if (k1 <= k2) continue;
        if (_o[k1]->l() != _o[k2]->l()) continue;
        int lidx = _lambdaMap[k1*100+k2];
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          if (_g->isLog())
            Sd(lidx) += matched[ir](k1)*matched[ir](k2)*std::pow((*_g)(ir), 2-1)*dr;
          else
            Sd(lidx) += matched[ir](k1)*matched[ir](k2)*std::pow((*_g)(ir), 2)*dr;
        }
      }
    }

    // recalculate the function being minimized, because we are changing energy for orbital k independently from the others
    VectorXld Fd = _irs.solve(EdE, l, Fmd, Kmd, matched, Rnodesd);
    if (k < _o.size()) {
      dPar(k) = 0;
      if (Fd(k) - Fn(k) != 0)
        dPar(k) = Fn(k)*dE[k]/(Fd(k) - Fn(k));
    } else {
      dPar(k) = 0;
      if (Sd(k-_o.size()) - Sn(k-_o.size()) != 0)
        dPar(k) = Sd(k-_o.size())*1e-2/(Sd(k-_o.size()) - Sn(k-_o.size()));
    }
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
  
      _dE[k] = -gamma*dPar(k);

      std::cout << "INFO: Orbital " << k << " (with secant method), dE = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
      if (_dE[k] > 0) {
        _Emin[k] = _o[k]->E();
      } else if (_dE[k] < 0) {
        _Emax[k] = _o[k]->E();
      }
    }
  }
  for (int k = 0; k < _lambda.size(); ++k) {
    _dlambda[k] = -gamma*dPar(_o.size()+k);
    std::cout << "INFO: Lagrange multiplier " << k << " (with secant method), dlambda = " << _dlambda[k] << " (probe dlambda = " << 1e-2 << ")" << std::endl;
  }

  VectorXld tmp;
  tmp.resize(_o.size());
  for (int k = 0; k < _o.size(); ++k) {
    tmp(k) = _o[k]->E();
  }
  _historyE.push_back(tmp);
  _historyF.push_back(Fn);
  VectorXld tmpl;
  tmpl.resize(_lambda.size());
  for (int k = 0; k < _lambda.size(); ++k) {
    tmpl(k) = _lambda[k];
  }
  _historyL.push_back(tmpl);

  return Fn.norm() + Sn.norm();
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
        ldouble dr = (*_g)(ir+1) - (*_g)(ir);
        if (_g->isLog())
          Sn(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2-1)*dr;
        else
          Sn(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2)*dr;
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
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matchedSt[idx][i];
      if (i >= 10 && i < _g->N() - 4 && (*_o[k])(i)*(*_o[k])(i-1) < 0) {
        _nodes[k] += 1;
        std::cout << "Orbital " << k << ": Found node at i=" << i << ", r = " << (*_g)(i) << std::endl;
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
      dE[k] = E[k]*1e-2/((ldouble) _o[k]->n());
      if (iterE >= 1) {
        dE[k] = _historyE[iterE-1][k] - _historyE[iterE][k];
      }
      if (dE[k] == 0)
        dE[k] = E[k]*1e-2/_o[k]->n();

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
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          if (_g->isLog())
            Sd(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2-1)*dr;
          else
            Sd(lidx) += matchedSt[k1][ir]*matchedSt[k2][ir]*std::pow((*_g)(ir), 2)*dr;
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
  dPar = J.inverse()*ParN;
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

  for (int k = 0; k < _o.size(); ++k) {
    int idx = _om.index(k);
    if (_nodes[k] < _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too few nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emin[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/((ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      _historyL.clear();
      _historyE.clear();
      _historyF.clear();
    } else if (_nodes[k] > _o[k]->n() - _o[k]->l() - 1) {
      std::cout << "INFO: Too many nodes in orbital " << k << ", skipping dE by large enough amount to go to the next node position." << std::endl;
      _Emax[k] = _o[k]->E();
      _dE[k] = -_o[k]->E() + (_Emin[k] + _Emax[k])*0.5;
      //_dE[k] = _o[k]->E()*(_nodes[k] - _o[k]->n() + _o[k]->l() + 1)/( (ldouble) 20.0*_o[k]->n());
      std::cout << "INFO: Orbital " << k << ", new dE = " << _dE[k] << std::endl;
      _historyL.clear();
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
      VectorXld tmpl;
      tmpl.resize(_lambda.size());
      for (int k = 0; k < _lambda.size(); ++k) {
        tmpl(k) = _lambda[k];
      }
      _historyE.push_back(tmp);
      _historyF.push_back(Fn);
      _historyL.push_back(tmpl);
    }
  }
  
  return Fn.squaredNorm()+Sn.squaredNorm();
}

