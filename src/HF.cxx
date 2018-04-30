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

ldouble HF::solveForFixedPotentials(int Niter, ldouble F0stop) {
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
ldouble HF::stepGordon(ldouble gamma) {
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
  std::vector<MatrixXld> Cmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, Cmn, E);

  ldouble Fn = _igs.solve(E, l, Fmn, Kmn, Cmn, matched);

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
    std::vector<MatrixXld> Cmd;
    calculateFMatrix(Fmd, Kmd, Cmd, EdE);

    ldouble Fd = _igs.solve(EdE, l, Fmd, Kmd, Cmn, matched);
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
ldouble HF::stepRenormalised(ldouble gamma) {
  int N = _om.N();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = 1e-3;
    E[k] = _o[k]->E();
    l[k] = _o[k]->l();
  }

  std::vector<MatrixXld> Fmn;
  std::vector<MatrixXld> Kmn;
  std::vector<MatrixXld> Cmn;
  std::vector<VectorXld> matched;
  calculateFMatrix(Fmn, Kmn, Cmn, E);

  ldouble Fn = _irs.solve(E, l, Fmn, Kmn, Cmn, matched);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matched[i](idx);
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
    std::vector<MatrixXld> Cmd;
    calculateFMatrix(Fmd, Kmd, Cmd, EdE);

    ldouble Fd = _irs.solve(EdE, l, Fmd, Kmd, Cmd, matched);
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
    //_dE[k] = -gamma*grad(k); // for root finding
    //if (std::fabs(_dE[k]) > 0.1) _dE[k] = 0.1*_dE[k]/std::fabs(_dE[k]);
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
ldouble HF::stepSparse(ldouble gamma) {
  // 1) build sparse matrix _A
  // 2) build sparse matrix _b
  _lsb.prepareMatrices(_A, _b0, _pot, _vd, _vex);
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
      if (i >= 10 && (*_g)(i) < std::pow(_o.size(),2) && i < _g->N() - 4 && (*_o[k])(i)*(*_o[k])(i-1) <= 0) {
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
ldouble HF::stepStandard(ldouble gamma) {
  int N = _om.N();

  std::vector<ldouble> E(_o.size(), 0);
  std::vector<int> l(_o.size(), 0);

  std::vector<ldouble> dE(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    dE[k] = 1e-3;
    E[k] = _o[k]->E();
    l[k] = _o[k]->l();
  }

  ldouble Fn = _iss.solve(E, _pot, _vd, _vex, matchedSt);

  for (int k = 0; k < _o.size(); ++k) {
    _nodes[k] = 0;
    int l = _o[k]->l();
    int m = _o[k]->m();
    int idx = _om.index(k);
    for (int i = 0; i < _g->N(); ++i) {
      (*_o[k])(i) = matchedSt[idx][i];
      if (i >= 10 && i < _g->N() - 4 && matchedSt[idx][i]*matchedSt[idx][i-1] <= 0) {
        _nodes[k] += 1;
        std::cout << "Orbital " << k << ": Found node at i=" << i << ", r = " << (*_g)(i) << std::endl;
      }
    }
  }

  std::vector<ldouble> grad(_o.size());
  for (int k = 0; k < _o.size(); ++k) {
    std::vector<ldouble> EdE = E;
    EdE[k] += dE[k];

    ldouble Fd = _iss.solve(EdE, _pot, _vd, _vex, matchedSt);

    grad[k] = (Fd - Fn)/dE[k];
  }

  ldouble F = Fn;
  for (int k = 0; k < _o.size(); ++k) {
    if (grad[k] != 0) {
      _dE[k] = Fn/grad[k]; // for root finding
      _dE[k] *= -gamma;
    } else {
      _dE[k] = 0;
    }
    if (std::fabs(_dE[k]) > 0.5) _dE[k] = 0.5*_dE[k]/std::fabs(_dE[k]);
    std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
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

  return F;
}


void HF::save(const std::string fout) {
  std::ofstream f(fout.c_str());
  f << std::setw(10) << "method" << " " << std::setw(10) << _method << std::endl;
  f << std::setw(10) << "Z" << " " << std::setw(10) << _Z << std::endl;
  f << std::setw(10) << "gamma_scf" << " " << std::setw(10) << _gamma_scf << std::endl;
  f << std::setw(10) << "central" << " " << std::setw(10) << 1 << std::endl;
  f << std::setw(10) << "grid.isLog" << " " << std::setw(10) << _g->isLog() << std::endl;
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
  _g->reset((bool) sr.getInt("grid.isLog"), sr.getDouble("grid.dx"), sr.getInt("grid.N"), sr.getDouble("grid.rmin"));
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
    //calculateY();
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
      _vexsum[std::pair<int, int>(k, k2)] = Vradial(_g->N(), 0);
    }
  }

  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();

    // calculate it first with filled orbitals, dividing by the number of orbitals
    // this is exact if all 2(2*l+1) orbitals in this level are filled
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      //if (k1 == k2) continue;
      if (_o[k1]->spin()*_o[k2]->spin() < 0) continue;

      int l2 = _o[k2]->l();
      int m2 = _o[k2]->m();
      std::cout << "Calculating Vex term from k1 = " << k1 << ", k2 = " << k2 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

      /*for (int k = 0; k <= 5; k += 1) {
        ldouble B = 0.0;
        if (k == 0 && l1 == 0 && l2 == 0) B = 1.0;
        if (k == 0 && l1 == 1 && l2 == 1) B = 1.0/3.0;

        //if (k == 0 && l1 == 0 && l2 == 0) B = 1.0;
        //if (k == 0 && l1 == 1 && l2 == 1 && m1 == m2) B = 1.0;
        //if (k == 0 && l1 == 2 && l2 == 2 && m1 == m2) B = 1.0;

        //if (k == 2 && l1 == 0 && l2 == 2 && m1 ==  0 && m2 ==  0) B =  0.1118033989;
        //if (k == 2 && l1 == 1 && l2 == 1 && m1 == -1 && m2 == -1) B = -0.05;
        //if (k == 2 && l1 == 1 && l2 == 1 && m1 ==  0 && m2 ==  0) B =  0.10;
        //if (k == 2 && l1 == 1 && l2 == 1 && m1 ==  1 && m2 ==  1) B = -0.05;

        //B = 1.0/((ldouble) (2*k + 1))*std::pow(CG(l1p, l2p, 0, 0, k, 0), 2);
        if (B == 0) continue;
        // This is the extra k parts
        for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
          ldouble r1 = (*_g)(ir1);
          _vexsum[std::pair<int,int>(k1, k2)][ir1] += B * _Y[10000*k + 100*k1 + 1*k2][ir1];
        }
      }*/

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
  for (int k = 0; k <= 6; ++k) {
    std::cout << "Calculating Y for k "<< k << std::endl;
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      int l1 = _o[k1]->l();
      int m1 = _o[k1]->m();
      for (int k2 = 0; k2 < _o.size(); ++k2) {
        int l2 = _o[k2]->l();
        int m2 = _o[k2]->m();
        _Y[10000*k + 100*k1 + 1*k2] = Vradial(_g->N(), 0);

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

      }
    }
  }
}

void HF::calculateVd(ldouble gamma) {
  std::cout << "Calculating Vd." << std::endl;

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vradial(_g->N(), 0);
  }

  // calculate it first with filled orbitals, dividing by the number of orbitals
  // this is exact if all 2(2*l+1) orbitals in this level are filled
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();
    std::cout << "Calculating Vd term from k = " << k1 << " (averaging over orbitals assuming filled orbitals)" << std::endl;

    //for (int k2 = 0; k2 < _o.size(); ++k2) {
    //  //if (k2 == k1) continue;
    //  // This is the T part (the rest of T is just Z/r)
    //  // The - Y_0 term is in Vex
    //  for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
    //    _vdsum[k1][ir1] += _Y[10000*0 + 100*k2 + 1*k2][ir1];
    //  }
    //}

    //for (int k = 2; k <= 6; k += 2) {
    //  ldouble A = 0.0;
    //  if (k == 2 && l1 == 1 && m1 == -1) A = -0.05;
    //  if (k == 2 && l1 == 1 && m1 == 0) A = 0.10;
    //  if (k == 2 && l1 == 1 && m1 == 1) A = -0.05;

    //  if (k == 2 && l1 == 2 && m1 == -2) A = -0.0714285714;
    //  if (k == 2 && l1 == 2 && m1 == -1) A = 0.0357142857;
    //  if (k == 2 && l1 == 2 && m1 == 0) A = 0.0714285714;
    //  if (k == 2 && l1 == 2 && m1 == 1) A = 0.0357142857;
    //  if (k == 2 && l1 == 2 && m1 == 2) A = -0.0714285714;

    //  if (k == 4 && l1 == 2 && m1 == -2) A =  0.0066964286;
    //  //if (k == 4 && l1 == 2 && m1 == -1) A =  0;
    //  //if (k == 4 && l1 == 2 && m1 ==  0) A =  0;
    //  //if (k == 4 && l1 == 2 && m1 ==  1) A =  0;
    //  if (k == 4 && l1 == 2 && m1 ==  2) A =  0.0066964286;

    //  if (A == 0) continue;
    //  // This is the extra k parts
    //  for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
    //    _vdsum[k1][ir1] += A * _Y[10000*k + 100*k1 + 1*k1][ir1];
    //  }
    //}

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




