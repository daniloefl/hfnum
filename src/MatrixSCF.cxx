#include "MatrixSCF.h"

#include <stdexcept>
#include <exception>

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

MatrixSCF::MatrixSCF() {
  _Z = 1;
  _b = 0;
  _o_up.clear();
  _o_dw.clear();
  _gamma_scf = 0.3;
}

void MatrixSCF::setBasis(Basis *b) {
  _b = b;
}

void MatrixSCF::setZ(ldouble Z) {
  _Z = Z;
}

ldouble MatrixSCF::Z() {
  return _Z;
}

MatrixSCF::~MatrixSCF() {
}

int MatrixSCF::getNOrbitals(int s) {
  if (s > 0) return _o_up.size();
  return _o_dw.size();
}

int MatrixSCF::getOrbital_n(int no, int s) {
  if (s > 0) return _o_up[no].n;
  return _o_dw[no].n;
}

std::string MatrixSCF::getOrbitalName(int no, int s) {
  OrbitalQuantumNumbers o;
  if (s > 0) o = _o_up[no];
  else o = _o_dw[no];

  std::string name = "";
  name += std::to_string(o.n);
  int l = o.l;
  int m = o.m;
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

ldouble MatrixSCF::getOrbital_E(int no, int s) {
  if (s > 0) _o_up[no].E;
  return _o_dw[no].E;
}

int MatrixSCF::getOrbital_l(int no, int s) {
  if (s > 0) _o_up[no].l;
  return _o_dw[no].l;
}

int MatrixSCF::getOrbital_m(int no, int s) {
  if (s > 0) _o_up[no].m;
  return _o_dw[no].m;
}

int MatrixSCF::getOrbital_s(int no, int s) {
  if (s > 0) _o_up[no].s;
  return _o_dw[no].s;
}

void MatrixSCF::gammaSCF(ldouble g) {
  _gamma_scf = g;
}


void MatrixSCF::addOrbital(int n, int l, int m, int s) {
  if (s > 0)
    _o_up.push_back(OrbitalQuantumNumbers(n, l, m, s, -_Z*_Z/std::pow(n, 2)*0.5));
  else
    _o_dw.push_back(OrbitalQuantumNumbers(n, l, m, s, -_Z*_Z/std::pow(n, 2)*0.5));
}

OrbitalQuantumNumbers::OrbitalQuantumNumbers(int in, int il, int im, int is, ldouble iE)
  : n(in), l(il), m(im), s(is), E(iE) {
}

OrbitalQuantumNumbers::OrbitalQuantumNumbers(const OrbitalQuantumNumbers &o)
  : n(o.n), l(o.l), m(o.m), s(o.s), E(o.E) {
}

OrbitalQuantumNumbers &OrbitalQuantumNumbers::operator =(const OrbitalQuantumNumbers &o) {
  n = o.n;
  l = o.l;
  m = o.m;
  s = o.s;
  E = o.E;
  return *this;
}

OrbitalQuantumNumbers::~OrbitalQuantumNumbers() {
}


