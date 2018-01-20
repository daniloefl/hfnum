#include "GTO.h"
#include <fstream>

GTO::GTO()
  : Basis() {
  _Z = 1;
}

GTO::~GTO() {
}

void GTO::setZ(ldouble Z) {
  _Z = Z;
}

void GTO::load(const std::string &fname) {
  std::ifstream f(fname.c_str());
  std::string line;
  while(std::getline(f, line)) {
    if (line == "") continue;
    if (line.at(0) == '#') continue;
    std::stringstream ss;
    ss.str(line);
    GTOUnit u;
    ss >> u.alpha >> u.n >> u.l >> u.m;
    _u.push_back(u);
  }
}

ldouble GTO::dot(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  // s integral
  if (_u[i].l != 0 || _u[j].l != 0) return 0;
  ldouble p = _u[i].alpha + _u[j].alpha;
  return std::pow(M_PI/p, 1.5);
}

ldouble GTO::T(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  // s integral
  if (_u[i].l != 0 || _u[j].l != 0) return 0;
  ldouble p = _u[i].alpha + _u[j].alpha;
  return 3*_u[i].alpha*_u[j].alpha/p*std::pow(M_PI/p, 1.5);
}

ldouble F0(ldouble t) {
  if (t == 0) return 0;
  return 0.5*std::sqrt(M_PI/t)*std::erf(std::sqrt(t));
}

ldouble GTO::V(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  // s integral
  if (_u[i].l != 0 || _u[j].l != 0) return 0;
  ldouble p = _u[i].alpha + _u[j].alpha;
  return -2*M_PI/p*_Z;
}

ldouble GTO::J(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
}

ldouble GTO::K(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
}

