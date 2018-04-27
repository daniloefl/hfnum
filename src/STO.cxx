#include "STO.h"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;

STO::STO()
  : Basis() {
  setZ(1);
}

STO::~STO() {
}

void STO::setZ(ldouble Z) {
  _Z = Z;
  _u.clear();
  STOUnit u;
  // u = 1/(sqrt( (2*n)! )) (2*xi)^(n+0.5) r^(n-1) exp(-xi r) Y_lm (theta, phi)
  double coeff[] = {1.0, 0.5, 2.0};
  for (size_t k = 0; k < sizeof(coeff)/sizeof(double); ++k) {
    u.xi = coeff[k];
    u.n = 0;
    u.l = 0;
    u.m = 0;
    _u.push_back(u);
  }
  std::cout << "Loaded STOs with exponents: ";
  for (auto &k : _u) {
    std::cout << k.xi << " ";
  }
  std::cout << std::endl;

}

ldouble STO::norm(int i) {
  const int l = _u[i].l;
  const int xi = _u[i].xi;
  if (l == 0) {
    return 1.0;
  } else if (l == 1) {
    return 1.0;
  }
  return 0; // This will cause a crash due to division by zero if a certain l is not implemented ---> this is what we want to inform the user something is deeply wrong
}

ldouble STO::value(int k, ldouble r, int l_proj, int m_proj) {
  if (l_proj != _u[k].l || m_proj != _u[k].m) return 0;
  return 1.0/sqrt(factorial(2*_u[k].n)) * pow(2*_u[k].xi, _u[k].n+0.5) * pow(r, _u[k].n-1)*exp(-_u[k].xi*r*r)*norm(k);
}

int STO::N() {
  return _u.size();
}

void STO::load(const std::string &fname) {
  _u.clear();
  std::ifstream f(fname.c_str());
  std::string line;
  while(std::getline(f, line)) {
    if (line == "") continue;
    if (line.at(0) == '#') continue;
    std::stringstream ss;
    ss.str(line);
    STOUnit u;
    ss >> u.xi >> u.n >> u.l >> u.m;
    _u.push_back(u);
  }
}

ldouble STO::dot(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  const ldouble xi1 = _u[i].xi;
  const ldouble xi2 = _u[j].xi;
  const int li1 = _u[i].l;
  const int mi1 = _u[i].m;
  const int li2 = _u[j].l;
  const int mi2 = _u[j].m;
  const int ni1 = _u[i].n;
  const int ni2 = _u[j].n;
  return 0;
}

ldouble STO::T(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  const ldouble xi1 = _u[i].xi;
  const ldouble xi2 = _u[j].xi;
  const int li1 = _u[i].l;
  const int mi1 = _u[i].m;
  const int li2 = _u[j].l;
  const int mi2 = _u[j].m;
  const int ni1 = _u[i].n;
  const int ni2 = _u[j].n;
  return 0;
}

ldouble STO::V(int i, int j) {
  if (i >= _u.size() || j >= _u.size()) return 0;
  const ldouble xi1 = _u[i].xi;
  const ldouble xi2 = _u[j].xi;
  const int li1 = _u[i].l;
  const int mi1 = _u[i].m;
  const int li2 = _u[j].l;
  const int mi2 = _u[j].m;
  const int ni1 = _u[i].n;
  const int ni2 = _u[j].n;
  return 0;
}

// 1/|ra - rb| = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1) r<^l/r>^(l+1) Y*lm(Oa) Ylm(Ob)
// <a2| <b1| 1/r12 |c1> |d2> = g
// f(2) = <b1|1/r12|c1> = int r^(nb+nc+2) exp(-(ab+ac) r^2) Y_b(O1) Y_c(O1) 1/r12 dr1 dO1 = sum_lm 4 pi / (2l+1) int r^(nb+nc+2) exp(-(ab+ac) r^2) r<^l/r>^(l+1) Y_b(O1) Y_c(O1) Y*lm(O1) dr1 dO1 Ylm(O2)
// g = sum_lm 4 pi / (2l+1) int_2 int_1 r1^(nb+nc+2) exp(-(ab+ac) r1^2) r<^l/r>^(l+1) Y_b(O1) Y_c(O1) Y*lm(O1) dr1 dO1 r2^(na+nd+2) exp(-(aa+ad) r2^2) Y_a(O2) Y_d(O2) Ylm(O2) dr2 dO2
//   = sum_lm 4 pi / (2l+1) [ int_2 [ int_1 r1^(nb+nc+2) exp(-(ab+ac) r1^2) r<^l/r>^(l+1) dr1 ] r2^(na+nd+2) exp(-(aa+ad) r2^2) dr2] [int_O1 Y_b(O1) Y_c(O1) Y*lm(O1) dO1] [int_O2 Y_a(O2) Y_d(O2) Ylm(O2) dO2 ]
// [int_O1 Y_b(O1) Y_c(O1) Y*lm(O1) dO1] = (-1)^m int_O1 Y_b(O1) Y_c(O1) Y_l,-m(O1) dO1 = std::sqrt((lb+1)*(lc+1)/(4*M_PI*(l+1)))*CG(lb, lc, 0, 0, l, 0)*CG(lb, lc, mb, mc, l, m)
// [int_O2 Y_a(O2) Y_d(O2) Ylm(O2) dO2] = int_O2 Y_a(O2) Y_d(O2) Y_lm(O2) dO1 = std::pow(-1, m)*std::sqrt((la+1)*(ld+1)/(4*M_PI*(l+1)))*CG(la, ld, 0, 0, l, 0)*CG(la, ld, ma, md, l, -m)
// int_1 r1^(nb+nc+2) exp(-(ab+ac) r1^2) r<^l/r>^(l+1) dr1 = int_r2^inf ... + int_0^r2 ...
//
// for r2 < r1:
// [ int_r2^inf r1^(nb+nc+2) exp(-(ab+ac) r1^2) r<^l/r>^(l+1) dr1 ] = int_r2^inf r1^(nb+nc+2) exp(-(ab+ac) r1^2) r2^l/r1^(l+1) dr1 =
//                                                             = r2^l int_r2^inf r1^(nb+nc+2-l-1) exp(-(ab+ac) r1^2) dr1
//                                                             = r2^l sqrt(2*(ab+ac))^(-nb-nc-2+l+1) int_r2^inf (  sqrt(2*(ab+ac)) r1)^(nb+nc+2-l-1) exp(- ( sqrt(2*(ab+ac)) r1)^2/2 ) dr1
//                                                             = r2^l sqrt(2*(ab+ac))^(-nb-nc-2+l+2) sqrt(2pi) int_(sqrt(2*(ab+ac)) r2)^inf 1/sqrt(2pi) x^(nb+nc+2-l-1) exp(- x^2/2 ) dx, with x = sqrt(2*(ab+ac)) r1
//                                  if nb+nc+2-l-1 even        = r2^l sqrt(2*(ab+ac))^(-nb-nc-2+l+2) sqrt(2pi) [ (nb+nc+2-l-2)!! + ...
// for r1 < r2:
// [ int_0^r2 r1^(nb+nc+2) exp(-(ab+ac) r1^2) r<^l/r>^(l+1) dr1 ] = int_0^r2 r1^(nb+nc+2) exp(-(ab+ac) r1^2) r1^l/r2^(l+1) dr1 =
//                                                             = r2^(-l-1) int_0^r2 r1^(nb+nc+2+l) exp(-(ab+ac) r1^2) dr1
// Very long solution ...
// 
// Useful:
// Y_a Y_b = sum_LM sqrt( (2la + 1) (2lb+1) / (2L+1) ) CG(la, lb, 0, 0, L, 0) CG(la, lb, ma, mb, L, M) Y_LM
//
// Try Fourier transform to avoid r< and r> ... 4 pi / (2pi)^3 int exp(i k.(r2 - r1))/k^2 dk = 1/r12
// <b1|1/r12|c1> = int r^(nb+nc+2) exp(-(ab+ac) r^2) Y_b(O1) Y_c(O1) 1/r12 dr1 dO1
//               = 4 pi / (2pi)^3 int_r int_k r^(nb+nc+2) exp(-(ab+ac) r^2) exp(i k . (r2 - r1))/k^2 Y_b(O1) Y_c(O1) dk dr1 dO1
//               = 4 pi / (2pi)^3 sum_LM sqrt( (2lb + 1) (2lc+1) / (2L+1) ) CG(lb, lc, 0, 0, L, 0) CG(lb, lc, mb, mc, L, M) int_r int_k r^(nb+nc+2) exp(-(ab+ac) r^2) exp(i k . (r2 - r1))/k^2 Y_LM(O1) dk dr1 dO1
// <a2| <b1| 1/r12 |c1> |d2> = 4 pi / (2pi)^3 int_r1 int_r2 int_k r1^(nb+nc+2) exp(-(ab+ac) r1^2) exp(i k . (r2 - r1))/k^2 Y_b(O1) Y_c(O1) r2^(na+nd+2) exp(-(aa+ad) r2^2) Y_a(O2) Y_d(O2) dk dr1 dO1 dr2 dO2
//                           = 4 pi / (2pi)^9 int_r1 int_r2
//                                            int_k exp(i k . (r2 - r1))/k^2 dk
//                                            int_k1 (i)^(nb+nc+2) deriv^(nb+nc+2) (exp (-k1^2 / ( 4 * (ab+ac)) )) sqrt(1/(2*(ab+ac))) exp(i k1 r1) dk1
//                                            int_k2 (i)^(na+nd+2) deriv^(na+nd+2) (exp (-k2^2 / ( 4 * (aa+ad)) )) sqrt(1/(2*(aa+ad))) exp(i k2 r2) dk2
//                                            Y_b(O1) Y_c(O1) Y_a(O2) Y_d(O2) dr1 dr2 --> ignoring this part for now to make it easier -> = 1/(4pi)^2
//                           = 1 / (2 pi)^9 1/(4pi) (i)^(nb+nc+2) (i)^(na+nd+2) sqrt(1/(2*(ab+ac))) sqrt(1/(2*(aa+ad))) int_r1 int_r2 int_k int_k1 int_k2 dr1 dr2 dk dk1 dk2
//                                            exp(i (k + k2) r2) exp (i (-k + k1) r1)
//                                            1/k^2 deriv^(nb+nc+2) (exp (-k1^2 / ( 4 * (ab+ac)) )) deriv^(na+nd+2) (exp (-k2^2 / ( 4 * (aa+ad)) ))
//                           = 1 / (2 pi)^9 1/(4pi) (i)^(nb+nc+2) (i)^(na+nd+2) sqrt(1/(2*(ab+ac))) sqrt(1/(2*(aa+ad))) int_k int_k1 int_k2 dk dk1 dk2
//                                            delta(k2+k) delta(k1-k)
//                                            1/k^2 deriv^(nb+nc+2) (exp (-k1^2 / ( 4 * (ab+ac)) )) deriv^(na+nd+2) (exp (-k2^2 / ( 4 * (aa+ad)) ))
//                           = 1 / (2 pi)^3 1/(4pi) (i)^(nb+nc+2) (-i)^(na+nd+2) sqrt(1/(2*(ab+ac))) sqrt(1/(2*(aa+ad)))  int_k dk
//                                            1/k^2 deriv^(nb+nc+2) (exp (-k^2 / ( 4 * (ab+ac)) )) deriv^(na+nd+2) (exp (-k^2 / ( 4 * (aa+ad)) ))
//                           = 1 / (2 pi)^3 1/(4pi) (i)^(nb+nc+2) (-i)^(na+nd+2) sqrt(1/(2*(ab+ac))) sqrt(1/(2*(aa+ad)))  int_k dk
//                                            1/k^2 deriv^(nb+nc+2) (exp (-k^2 / ( 4 * (ab+ac)) )) deriv^(na+nd+2) (exp (-k^2 / ( 4 * (aa+ad)) ))
//
// Complicated to deal with spherical harmonics ... move to a different basis
ldouble STO::ABCD(int a, int b, int c, int d) {
  if (a >= _u.size() || b >= _u.size() || c >= _u.size() || d >= _u.size()) return 0;
  const ldouble xi_a = _u[a].xi;
  const ldouble xi_b = _u[b].xi;
  const ldouble xi_c = _u[c].xi;
  const ldouble xi_d = _u[d].xi;
  const int l_a = _u[a].l;
  const int m_a = _u[a].m;
  const int l_b = _u[b].l;
  const int m_b = _u[b].m;
  const int l_c = _u[c].l;
  const int m_c = _u[c].m;
  const int l_d = _u[d].l;
  const int m_d = _u[d].m;
  // not estimated yet in p or d states
  if (l_a == 0 && l_b == 0 && l_c == 0 && l_d == 0) {
    return 2*std::pow(M_PI, 2.5)/( (xi_a + xi_b) * (xi_c + xi_d) * std::sqrt(xi_a + xi_b + xi_c + xi_d) ) * F0( (xi_a + xi_b)*(xi_c + xi_d)/(xi_a + xi_b + xi_c + xi_d) )*norm(a)*norm(b)*norm(c)*norm(d);
  }
  return 0;
}


