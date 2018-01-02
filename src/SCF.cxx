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

void SCF::centralPotential(bool central) {
  _central = central;
}

SCF::SCF(const std::string fname)
  : _g(new Grid(true, 1e-1, 10, 1e-3)), _Z(1), _om(*_g, _o), _lsb(*_g, _o, icl, _om), _irs(*_g, _o, icl, _om), _igs(*_g, _o, icl, _om) {
  _own_grid = true;
  _pot.resize(_g->N());
  _central = true;
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
  _gamma_scf = 0.5;
  _method = 2;
  load(fname);
}

SCF::SCF()
  : _g(new Grid(true, 1e-1, 10, 1e-3)), _Z(1), _om(*_g, _o), _lsb(*_g, _o, icl, _om), _irs(*_g, _o, icl, _om), _igs(*_g, _o, icl, _om) {
  _own_grid = true;
  _pot.resize(_g->N());
  _central = true;
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
  _gamma_scf = 0.5;
  _method = 2;
}

Grid &SCF::getGrid() {
  return *_g;
}

python::list SCF::getR() const {
  return _g->getR();
}

void SCF::resetGrid(bool isLog, ldouble dx, int N, ldouble rmin) {
  _g->reset(isLog, dx, N, rmin);
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

void SCF::save(const std::string fout) {
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
  for (auto &i : _vd) {
    const int &k = i.first;
    Vd &v = i.second;
    for (auto &vr : v) {
      const std::pair<int, int> &lm = vr.first;
      const std::vector<ldouble> &vradial = vr.second;
      f << std::setw(10) << "vd" << " " << std::setw(10) << k;
      f << " " << std::setw(5) << "l" << " " << std::setw(5) << lm.first;
      f << " " << std::setw(5) << "m" << " " << std::setw(5) << lm.second;
      f << " " << std::setw(5) << "value";
      for (int ir = 0; ir < vradial.size(); ++ir) {
        f << " " << std::setw(64) << std::setprecision(60) << vradial[ir];
      }
      f << std::endl;
    }
  }
  for (auto &i : _vex) {
    const int &k1 = i.first.first;
    const int &k2 = i.first.second;
    Vd &v = i.second;
    for (auto &vr : v) {
      const std::pair<int, int> &lm = vr.first;
      const std::vector<ldouble> &vradial = vr.second;
      f << std::setw(10) << "vex" << " " << std::setw(10) << k1 << " " << std::setw(10) << k2;
      f << " " << std::setw(5) << "l" << " " << std::setw(5) << lm.first;
      f << " " << std::setw(5) << "m" << " " << std::setw(5) << lm.second;
      f << " " << std::setw(5) << "value";
      for (int ir = 0; ir < vradial.size(); ++ir) {
        f << " " << std::setw(64) << std::setprecision(60) << vradial[ir];
      }
      f << std::endl;
    }
  }
}

void SCF::load(const std::string fin) {
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
    } else if (mode == "vd") {
      int io;
      ss >> io;

      std::string trash;

      int v_l, v_m;
      ss >> trash >> v_l >> trash >> v_m;

      ss >> trash;

      _vd[io][std::pair<int, int>(v_l, v_m)] = std::vector<ldouble>(g_N, 0);
      ldouble read_value;
      for (int k = 0; k < g_N; ++k) {
        ss >> read_value;
        _vd[io][std::pair<int, int>(v_l, v_m)][k] = read_value;
      }
    } else if (mode == "vex") {
      int io1, io2;
      ss >> io1 >> io2;

      std::string trash;

      int v_l, v_m;
      ss >> trash >> v_l >> trash >> v_m;

      ss >> trash;

      _vex[std::pair<int, int>(io1, io2)][std::pair<int, int>(v_l, v_m)] = std::vector<ldouble>(g_N, 0);
      ldouble read_value;
      for (int k = 0; k < g_N; ++k) {
        ss >> read_value;
        _vex[std::pair<int, int>(io1, io2)][std::pair<int, int>(v_l, v_m)][k] = read_value;
      }
    }
  }
  std::cout << "Load resetting grid with isLog = " << g_isLog << ", dx = " << g_dx << ", g_N = " << g_N << ", g_rmin = " << g_rmin << std::endl;
  _g->reset(g_isLog, g_dx, g_N, g_rmin);
  _pot.resize(_g->N());
  for (int k = 0; k < _g->N(); ++k) {
    _pot[k] = -_Z/(*_g)(k);
  }
}

std::vector<ldouble> SCF::getDirectPotential(int k) {
  return _vd[k][std::pair<int, int>(_o[k]->initialL(), _o[k]->initialM())];
}

std::vector<ldouble> SCF::getExchangePotential(int k, int k2) {
  return _vex[std::pair<int,int>(k,k2)][std::pair<int, int>(_o[k2]->initialL(), _o[k2]->initialM())];
}


python::list SCF::getDirectPotentialPython(int k) {
  python::list l;
  std::vector<ldouble> v = getDirectPotential(k);
  for (int k = 0; k < _g->N(); ++k) l.append(v[k]);
  return l;
}

python::list SCF::getExchangePotentialPython(int k, int k2) {
  python::list l;
  std::vector<ldouble> v = getExchangePotential(k, k2);
  for (int k = 0; k < _g->N(); ++k) l.append(v[k]);
  return l;
}


int SCF::getNOrbitals() {
  return _o.size();
}

int SCF::getOrbital_n(int no) {
  return _o[no]->initialN();
}

std::string SCF::getOrbitalName(int no) {
  std::string name = "";
  name += std::to_string(_o[no]->initialN());
  int l = _o[no]->initialL();
  int m = _o[no]->initialM();
  int s = _o[no]->spin();
  if (l == 0) name += "s";
  else if (l == 1) name += "p";
  else if (l == 2) name += "d";
  else if (l == 3) name += "f";
  else if (l == 4) name += "g";
  else if (l == 5) name += "h";
  else name += "?";
  name += "_{m=";
  name += std::to_string(m);
  name += "}";
  if (s > 0)
    name += "^+";
  else
    name += "^-";
  return name;
}

ldouble SCF::getOrbital_E(int no) {
  return _o[no]->E();
}

int SCF::getOrbital_l(int no) {
  return _o[no]->initialL();
}

int SCF::getOrbital_m(int no) {
  return _o[no]->initialM();
}

int SCF::getOrbital_s(int no) {
  return _o[no]->spin();
}

void SCF::method(int m) {
  if (m < 0) m = 0;
  if (m > 2) m = 2;
  _method = m;
}

python::list SCF::getNucleusPotentialPython() {
  python::list l;
  std::vector<ldouble> v = getNucleusPotential();
  for (int k = 0; k < _g->N(); ++k) l.append(v[k]);
  return l;
}
std::vector<ldouble> SCF::getOrbital(int no, int lo, int mo) {
  Orbital *o = _o[no];
  std::vector<ldouble> res;
  for (int k = 0; k < _g->N(); ++k) {
    res.push_back(o->getNorm(k, lo, mo, *_g));
  }
  return res;
}

std::vector<ldouble> SCF::getOrbitalCentral(int no) {
  Orbital *o = _o[no];
  std::vector<ldouble> res;
  for (int k = 0; k < _g->N(); ++k) {
    res.push_back(o->getNorm(k, o->initialL(), o->initialM(), *_g));
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
ldouble SCF::stepGordon(ldouble gamma) {
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
ldouble SCF::stepRenormalised(ldouble gamma) {
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

  /*
  // calculate Hessian(F) * dX = - grad(F)
  // this does not work if looking for an asymptotic minimum!
  // H(k1, k2) = [ (F(E+dE2+dE1) - F(E+dE2))/dE1 - (F(E+dE1) - F(E))/dE1 ]/dE2
  MatrixXld H(_o.size(), _o.size());
  H.setZero();
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    ldouble der1 = grad(k1); // derivative at F(E) w.r.t. E1
    // move E(k2) by delta E(k2) and calculate derivative there
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      std::vector<ldouble> EdE = E;
      EdE[k2] += dE[k2]; // move to this other point

      std::vector<MatrixXld> Fmd;
      std::vector<MatrixXld> Kmd;
      calculateFMatrix(Fmd, Kmd, EdE);
      ldouble Fd21 = _irs.solve(EdE, l, Fmd, Kmd, matched); // nominal at E(k2)+dE(k2)

      EdE[k1] += dE[k1]; // variation due to E(k1)

      calculateFMatrix(Fmd, Kmd, EdE);
      ldouble Fd22 = _irs.solve(EdE, l, Fmd, Kmd, matched);

      ldouble der2 = (Fd22 - Fd21)/(dE[k1]); // derivative at E(k2)+dE(k2)

      H(k1, k2) = (der2 - der1)/dE[k2];
    }
  }
  std::cout << "Hessian: " << std::endl << H << std::endl;
  std::cout << "grad: " << std::endl << grad << std::endl;
  VectorXld dX(_o.size());
  dX.setZero();
  JacobiSVD<MatrixXld> decH(H, ComputeThinU | ComputeThinV);
  if (H.determinant() != 0) {
    dX = decH.solve(-grad);
  }
  */



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
    std::cout << "Orbital " << k << ", dE(Jacobian) = " << _dE[k] << " (probe dE = " << dE[k] << ")" << std::endl;
  }

  return F;
}

// solve for a fixed energy and calculate _dE for the next step
ldouble SCF::stepSparse(ldouble gamma) {
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

void SCF::addOrbital(Orbital *o) {
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
  _vd.clear();
  _vex.clear();
  for (int k = 0; k < _o.size(); ++k) {
    _vd[k] = Vd();
    _vd[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vd[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
    _vd[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vd[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vex[std::pair<int, int>(k, k2)] = Vex();
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
      _vex[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
    }
  }

  for (int k = 0; k < _o.size(); ++k) {
    _vdsum[k] = Vd();
    _vdsum[k][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
    _vdsum[k][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      _vexsum[std::pair<int, int>(k, k2)] = Vex();
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(0, 0)] = std::vector<ldouble>(_g->N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, -1)] = std::vector<ldouble>(_g->N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 0)] = std::vector<ldouble>(_g->N(), 0);
      _vexsum[std::pair<int, int>(k, k2)][std::pair<int, int>(1, 1)] = std::vector<ldouble>(_g->N(), 0);
    }
  }
}



