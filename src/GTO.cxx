#include "GTO.h"
#include <fstream>
#include <iostream>

GTO::GTO()
  : Basis() {
  setZ(1);
}

GTO::~GTO() {
}

void GTO::setZ(ldouble Z) {
  _Z = Z;
  _u.clear();
  GTOUnit u;
  // 1/sqrt(norm) r^n exp(-alpha r^2) Y_{lm}
  // norm = int_0^inf r^2 r^n exp(-alpha r^2) dr = Gamma( (2+n+1)/2 )/ ( 2* alpha^((2+n+1)/2) )
  // Y_{lm} assumed normalised spherical harmonics, so int |Y_lm|^2 dOmega = 1
  for (int l = 0; l < 2; ++l) {
    for (int m = -l; m <= l; ++m) {
      for (int k = 0; k < 20; ++k) {
        u.alpha = std::exp(std::log(0.01) + k);
        u.n = l;
        u.l = l;
        u.m = m;
        _u.push_back(u);
      }
    }
  }
}

ldouble GTO::value(int k, ldouble r, int l_proj, int m_proj) {
  if (_u[k].l != l_proj || _u[k].m != m_proj) return 0;
  return std::pow(r, _u[k].n)*std::exp(-_u[k].alpha*r*r)/std::sqrt(std::tgamma((2.0 + _u[k].n + 1.0)/2.0)/(2*std::pow(_u[k].alpha, (2.0 + _u[k].n + 1.0)/2.0)));
}

int GTO::N() {
  return _u.size();
}

void GTO::load(const std::string &fname) {
  _u.clear();
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

// int_0^inf exp(-a x^2) dx = 0.5 sqrt(pi/a)
// d/da ( exp (-a x^2) ) = -x^2 exp(-a x^2)
// int_0^inf x^2 exp(-a x^2) dx = - int_0^inf d/da [exp (-a x^2) ] dx
//                              = - d/da int_0^inf exp (-a x^2) dx
//                              = - 0.5 sqrt(pi) d/da [a^-0.5]
//                              = 1/4 sqrt(pi/(a^3)) = 1/(4pi) sqrt((pi/a)^3)
// normalisation factor = 1/sqrt(
// <i|j> = int r^2 exp(-(ai+aj) r^2) Y_i Y_j dr dOmega
//       = 1/(4pi) sqrt((pi/(ai+aj))^3) delta(m_i - m_j) delta (l_i - l_j)
ldouble GTO::dot(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  return RPower(0, i, j);
}

// int_0^inf x^n exp(-a x^2) dx = Gamma((n+1)/2)/(2 a^((n+1)/2))
// calculate int_0^inf x^2 x^n x^li x^lj exp(-ai x^2) exp(-aj x^2) dx
ldouble GTO::RPower(int power, int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  if (_u[i].l != _u[j].l) return 0;
  if (_u[i].m != _u[j].m) return 0;
  ldouble p = _u[i].alpha + _u[j].alpha;
  ldouble c = (power + 2.0 + _u[i].n + _u[j].n + 1.0)/2.0;
  ldouble norm = std::tgamma((2.0 + _u[i].n + 1.0)/2.0)/(2*std::pow(_u[i].alpha, (2.0 + _u[i].n + 1.0)/2.0))*std::tgamma((2.0 + _u[j].n + 1.0)/2.0)/(2*std::pow(_u[j].alpha, (2.0 + _u[j].n + 1.0)/2.0));
  return std::tgamma(c)/(2*std::pow(p, c))/std::sqrt(norm);
}

// int_0^inf x^2 exp(-a x^2) dx = 1/(4pi) sqrt((pi/a)^3)
// d/da ( x^2 exp (-a x^2) ) = -x^4 exp(-a x^2)
// int_0^inf x^4 exp(-a x^2) dx = - int_0^inf d/da [x^2 exp (-a x^2) ] dx
//                              = - d/da int_0^inf x^2 exp (-a x^2) dx
//                              = - 1/(4pi) d/da sqrt((pi/a)^3)
//                              = 3/(8pi) sqrt(pi^3/a^5)
//                              = 3/(8 pi^2) sqrt((pi/a)^5)
// <i|r^2|j> = int r^4 exp(-(ai+aj) r^2) Y_i Y_j dr dOmega
//       = 3/(8 pi^2) sqrt((pi/(ai+aj))^5) delta(m_i - m_j) delta (l_i - l_j)
ldouble GTO::R2(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  return RPower(2, i, j);
}

// int_0^inf x exp(-a x^2) dx = 1/2 int_0^inf exp(-a x^2) d(x^2)
//                            = - 1/(2a) int_0^-inf exp(u) du
//                            = - 1/(2a) [ exp(-inf) - exp(0) ]
//                            = 1/(2a)
// d/da ( x exp (-a x^2) ) = -x^3 exp(-a x^2)
// int_0^inf x^3 exp(-a x^2) dx = - int_0^inf d/da [x exp (-a x^2) ] dx
//                              = - d/da int_0^inf x exp (-a x^2) dx
//                              = - 1/2 a^(-2)
// <i|r|j> = int r^3 exp(-(ai+aj) r^2) Y_i Y_j dr dOmega
//       = -1/2 (ai+aj)^(-2) delta(m_i - m_j) delta (l_i - l_j)
ldouble GTO::R(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  return RPower(1, i, j);
}

// int_0^inf exp(-a x^2) dx = 0.5 sqrt(pi/a)
// <i|1/r^2|j> = int exp(-(ai+aj) r^2) Y_i Y_j dr dOmega
//       = 1/2 sqrt(pi/(ai+aj)) delta(m_i - m_j) delta (l_i - l_j)
ldouble GTO::RMinus2(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  return RPower(-2, i, j);
}

// int_0^inf x exp(-a x^2) dx = 1/(2a)
// <i|1/r|j> = int r exp(-(ai+aj) r^2) Y_i Y_j dr dOmega
//       = 1/(2 (ai+aj)) delta(m_i - m_j) delta (l_i - l_j)
ldouble GTO::RMinus1(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  return RPower(-1, i, j);
}

// T|j> = -0.5 del^2 exp(-aj r^2) Y_j
//      = -0.5 { 1/r^2 Y_j d/dr [ r^2 d/dr (exp -aj r^2) ] - lj (lj+1)/r^2 |j> }
//      = -0.5 { Y_j 1/r^2 d/dr [ -2 aj r^3 exp(-aj r^2) ] - lj (lj+1)/r^2 |j> } 
//      = -0.5 { Y_j (-2 aj) 1/r^2 [ 3 r^2 exp(-aj r^2) - 2 aj r^4 exp (-aj r^2) ] - lj (lj+1)/r^2 |j> }
//      = -0.5 { -6 aj + 4 aj^2 r^2 - lj (lj+1)/r^2 } |j>
//      = [3 aj - 2 aj^2 r^2 + lj (lj+1) / (2 r^2) ] |j>
// <i|T|j> = 3 aj <i|j> - 2 aj^2 <i|r^2|j> + lj (lj+1)/2 <i|1/r^2|j>
// for l = 0, m = 0, and multiplying by 4pi
//         = 3 aj pi^(3/2) (ai+aj)^(-3/2) - 3/pi aj^2 sqrt((pi/(ai+aj))^5)
//         = 3 pi^(3/2) aj (ai+aj)^(-3/2) - 3 pi^(3/2) aj^2 (ai+aj)^(-5/2)
//         = [ 3 aj - 3 aj^2/(ai+aj) ] [pi/(ai+aj)]^(3/2)
//         = [ 3 ai aj + 3 aj^2 - 3 aj^2]/(ai+aj) [pi/(ai+aj)]^(3/2)
//         = 3 ai aj/(ai+aj) [pi/(ai+aj)]^(3/2)
ldouble GTO::T(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  ldouble aj = _u[j].alpha;
  ldouble lj = _u[j].l;
  return 3*aj*dot(i, j) - 2*std::pow(aj, 2)*R2(i, j) + 0.5*lj*(lj+1)*RMinus2(i, j);
}

// -Z<i|1/r|j> =
//             = - Z/(2 (ai+aj)) delta(m_i - m_j) delta (l_i - l_j)
// multiplying by 4 pi and for l = m = 0:
//             = - 2 pi Z/(ai+aj)
ldouble GTO::V(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  return -_Z*RMinus1(i, j);
}

// <a2| <b1| 1/r12 |c1> |d2> =
ldouble GTO::ABCD(int a, int b, int c, int d) {
  if (a >= _u.size() || b >= _u.size() || c >= _u.size() || d >= _u.size()) return 0;
  return 0;
}


