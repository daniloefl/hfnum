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
  : _g(new Grid(true, 1e-1, 10, 1e-3)), _Z(1), _om(*_g, _o), _lsb(*_g, _o, icl, _om), _irs(*_g, _o, icl, _om), _iss(*_g, _o, icl, _om), _igs(*_g, _o, icl, _om) {
  _own_grid = true;
  _pot.resize(_g->N());
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

