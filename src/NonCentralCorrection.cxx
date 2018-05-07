#include "NonCentralCorrection.h"
#include "StateReader.h"
#include "Orbital.h"
#include "Grid.h"
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <iostream>

#include <Eigen/Eigenvalues> 

NonCentralCorrection::NonCentralCorrection() {
}

NonCentralCorrection::~NonCentralCorrection() {
}

void NonCentralCorrection::load(const std::string &fname) {
  StateReader sr(fname);

  _o.clear();
  for (auto &o : _o) {
    delete o;
  }

  _Z = sr.getDouble("Z");
  _g->reset((gridType) sr.getInt("grid.isLog"), sr.getDouble("grid.dx"), sr.getInt("grid.N"), sr.getDouble("grid.rmin"));
  for (int k = 0; k < sr._o.size(); ++k) {
    _o.push_back(new Orbital(*sr.getOrbital(k)));
  }
  for (auto &k : sr._vd) {
    _vd[k.first] = k.second;
  }
  for (auto &k : sr._vex) {
    _vex[k.first] = k.second;
  }
}

void NonCentralCorrection::calculateY() {
  std::cout << "Calculating Y" << std::endl;
  // Calculating Y_k(orb1, orb2)[r]
  // index in Y is 10000*k + 100*orb1 + orb2
  for (int k = 0; k <= 2; ++k) {
    std::cout << "Calculating Y for k "<< k << std::endl;
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      int l1 = _o[k1]->l();
      int m1 = _o[k1]->m();
      for (int k2 = 0; k2 < _o.size(); ++k2) {
        int l2 = _o[k2]->l();
        int m2 = _o[k2]->m();
        _Y[10000*k + 100*k1 + 1*k2] = Vradial(_g->N(), 0);
        _Zt[10000*k + 100*k1 + 1*k2] = Vradial(_g->N(), 0);

        // r Y(r) = int_0^r Pk1*t Pk2*t (t/r)^k dt +
        //         int_r^inf Pk1*t Pk2*t (r/t)^(k+1) dt
        // Z(r) = int_0^r Pk1*t Pk2*t (t/r)^k dt
        // dZ/dr = Pk1*r Pk2*r - k/r Z
        // d(rY)/dr = 1/r [ (k+1) (rY) - (2k + 1) Z ]
        // Z (r=0) = 0
        // lim Y when r -> infinity = Z
        // define r = exp(x), x = ln(r)
        // dZ/dx = dZ/dr dr/dx = Pk1*r Pk2*r * r - k Z
        // d(rY)/dx = [ (k+1) (rY) - (2k + 1) Z ]
        // d(exp(kx)*Z)/dx = k exp(kx) Z + exp(kx) dZ/dx
        //                 = exp(kx) *Pk1*r Pk2 *r *r
        // d(exp(- (k+1)x) (rY))/dx = -(k+1) exp(-(k+1)x) (rY) + exp(-(k+1)x) d(rY)/dx
        //                       = - (2k +1) Z exp(-(k+1)x)
        _Zt[10000*k + 100*k1 + 1*k2][0] = 0;
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble rp1 = (*_g)(ir+1);
          ldouble x = std::log(r);
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble dx = std::log((*_g)(ir+1)) - std::log((*_g)(ir));
          ldouble fn = std::pow(r, 3)*_o[k1]->getNorm(ir, *_g) * _o[k2]->getNorm(ir, *_g);
          ldouble fnp1 = std::pow(rp1, 3)*_o[k1]->getNorm(ir+1, *_g) * _o[k2]->getNorm(ir+1, *_g);
          _Zt[10000*k + 100*k1 + 1*k2][ir+1] = std::exp(-dx*k)*_Zt[10000*k + 100*k1 + 1*k2][ir] + 0.5*(fnp1+fn)*std::exp(dx*k)*dx;
        }
        _Y[10000*k + 100*k1 + 1*k2][_g->N()-1] = _Zt[10000*k + 100*k1 + 1*k2][_g->N()-1];
        for (int ir = _g->N()-1; ir >= 1; --ir) {
          ldouble r = (*_g)(ir);
          ldouble x = std::log(r);
          ldouble dr = (*_g)(ir) - (*_g)(ir-1);
          ldouble dx = std::log((*_g)(ir)) - std::log((*_g)(ir-1));
          ldouble fn = (2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir];
          ldouble fnm1 = (2*k+1)*_Zt[10000*k + 100*k1 + 1*k2][ir-1];
          _Y[10000*k + 100*k1 + 1*k2][ir-1] = std::exp(-dx*(k+1))*_Y[10000*k + 100*k1 + 1*k2][ir] + 0.5*(fn+fnm1)*std::exp(-(k+1)*dx)*dx;
        }
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          _Y[10000*k + 100*k1 + 1*k2][ir] = _Y[10000*k + 100*k1 + 1*k2][ir]/r;
        }

        // classical integration:
        /*
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
        */

      }
    }
  }
}

void NonCentralCorrection::correct() {
  _Ec.resize(_o.size(), 0);
  for (auto &e : _Ec) e = 0;

  _c.resize(_o.size(), _o.size());
  _c.setZero();

  _J.resize(_o.size(), _o.size());
  _K.resize(_o.size(), _o.size());
  _J.setZero();
  _K.setZero();

  _Jcorr.resize(_o.size(), _o.size());
  _Kcorr.resize(_o.size(), _o.size());
  _Jcorr.setZero();
  _Kcorr.setZero();

  MatrixXcld cdeg(_o.size(), _o.size());
  cdeg.setZero();

  calculateY();

  int idx = 0;
  

  // find degeneracy
  std::vector<ldouble> E;
  std::vector<std::vector<int> > deg;
  for (int k = 0; k < _o.size(); ++k) {
    int deg_idx = -1;
    for (int j = 0; j < E.size(); ++j) {
      if (std::fabs(E[j] - _o[k]->E()) < 1e-3) {
        deg_idx = j;
      }
    }
    if (deg_idx < 0) {
      E.push_back(_o[k]->E());
      deg.push_back(std::vector<int>(1, k));
    } else {
      deg[deg_idx].push_back(k);
    }
  }

  for (int deg_idx = 0; deg_idx < E.size(); ++deg_idx) {
    std::cout << "Assuming these orbitals are in a degenerate level: ";
    for (auto &o : deg[deg_idx]) std::cout << " " << o;
    std::cout << std::endl;
  }

  MatrixXld dH(_o.size(), _o.size());
  dH.setZero();

  MatrixXld S(_o.size(), _o.size());
  S.setZero();

  int lmax = 2; // approximated here
  // define delta V = (full Vd - full Vex) - (Vd - Vex)
  // this is defined separately for each orbital equation: we want the error in the eigenenergies
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    lm tlm_d1(_o[k1]->l(), _o[k1]->m());
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      lm tlm_d2(_o[k2]->l(), _o[k2]->m());
      std::cout << "Calculating matrix elements for element " << k1 << ", " << k2 << std::endl;

      std::cout << "Calculating matrix element <" << k1 << "|S|" << k2 << ">" << std::endl;
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() >= 0) {
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble rp1 = (*_g)(ir+1);
          ldouble dr = 0;
          if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble fn = _o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2);
          ldouble fnp1 = _o[k1]->getNorm(ir+1, *_g)*_o[k2]->getNorm(ir+1, *_g)*std::pow(rp1, 2);
          S(k1, k2) += 0.5*(fn+fnp1)*dr;
        }
      }

      std::cout << "Calculating matrix element <" << k1 << "|Vd_old|" << k2 << ">" << std::endl;
      // now calculate dH, where the effect of all degenerate states is considered
      // the current inaccurate result follows
      // remember, in the central approximation: <r,s|vd(k1)|u_k1(r),l1,m1,s1> = vd(k1) * u_k1(r) Y_l1,m1(phi, theta) * delta(s == s1)
      // the angular and spin parts are the ones of k1, which is the orbital equation on which this operator acts
      // this is obviously wrong, but it is part of the assumption in the central approximation in HF.cxx
      // here we calculate <u_k2,l2,m2,s2|vd(k1)|u_k1,l1,m1,s1> = int_r u_k2(r) Vd(k1,ko)(r) u_k1(r) r^2 dr delta(l1 == l2, m1 == m2, s1 == s2)
      // - vd
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() >= 0) {
        ldouble A = 1;
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble rp1 = (*_g)(ir+1);
          ldouble dr = 0;
          if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble fn = A*_vd[k1][ir]*_o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2);
          ldouble fnp1 = A*_vd[k1][ir+1]*_o[k1]->getNorm(ir+1, *_g)*_o[k2]->getNorm(ir+1, *_g)*std::pow(rp1, 2);
          dH(k1, k2) += -0.5*(fn+fnp1)*dr;
        }
      }

      std::cout << "Calculating matrix element <" << k1 << "|Vex_old|" << k2 << ">" << std::endl;
      // + vex
      // remember, in the central approximation: <r,spin|vex(k1,ko)|u_k1(r),l,m,s> = vex(k1,ko) * u_ko(r) Y_k1(phi, theta) * delta(spin == spin_k1)
      // the angular and spin parts are the ones of k1, which is the orbital equation on which this operator acts
      // this is obviously wrong, but it is part of the assumption in the central approximation in HF.cxx
      // to correct for it the vex calculation in the end of this function treats the angular component correctly, but to calculate the correction, the incorrect
      // component needs to be calculated here and subtracted later, so that dH = variation in the Hamiltonian is consistent
      // here we calculate <u_k2,l2,m2,s2|vex(k1,ko)|u_k1,l1,m1,s1> = sum_ko int_r u_k2(r) Vex(k1,ko)(r) u_ko(r) r^2 dr delta(l1 == l2, m1 == m2, s1 == s2)
      // as we know vex(k1,ko)(r) == 0 if s1 != so, so we can use this simplification to speed things up
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k2]->spin()*_o[k1]->spin() >= 0) {
        for (int ko = 0; ko < _o.size(); ++ko) {
          ldouble A = 1;
          lm tlmo(_o[ko]->l(), _o[ko]->m());
          if (_o[ko]->spin()*_o[k1]->spin() < 0) continue; // vex == 0 here
          for (int ir = 0; ir < _g->N()-1; ++ir) {
            ldouble r = (*_g)(ir);
            ldouble rp1 = (*_g)(ir+1);
            ldouble dr = 0;
            if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
            ldouble fn = A*_vex[std::pair<int, int>(k1, ko)][ir]*_o[ko]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2);
            ldouble fnp1 = A*_vex[std::pair<int, int>(k1, ko)][ir+1]*_o[ko]->getNorm(ir+1, *_g)*_o[k2]->getNorm(ir+1, *_g)*std::pow(rp1, 2);
            dH(k1, k2) += 0.5*(fn+fnp1)*dr;
          }
        }
      }


      std::cout << "Calculating matrix element <" << k1 << "|Vd|" << k2 << ">" << std::endl;
      // + full vd
      // vd(r1) = sum_ko int |psi_ko(r2) Y_ko(Omega2)|^2 1/|r1 - r2| dOmega2 r2^2 dr2
      // dH(a,b) term will be: Term = sum_ko [int dr1 dOmega1 r1^2 psi_a(r1) psi_b(r1) Y*_a(O1) Y_b(O1) { int dr2 dOmega2 r2^2 psi_ko(r2)^2 Y_ko(Omega2)^2 1/|r1 - r2| } ]
      // Term = sum_ko int dr1 int dOmega1 int dr2 int dOmega2 [ r1^2 psi_a(r1) psi_b(r1) Y*_a(O1) Y_b(O1) r2^2 psi_ko(r2)^2 Y_ko(Omega2)^2 1/|r1 - r2| ]
      // 1/|r1 - r2| = sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) r_<^l/r_>^(l+1) Y_lm*(Omega1) Y_lm(Omega2)
      // Term = sum_ko int dr1 int dOmega1 int dr2 int dOmega2 sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_a(r1) psi_b(r1) psi_ko(r2)^2 Y*_a(O1) Y_b(O1) |Y_ko(O2)|^2 Y_lm(O1) Y_lm*(O2) ]
      // int dO1 Y*_a(O1) Y_b(O1) Y_lm*(O1) = (-1)^(m+m_a) int dO1 Y_(l_a, -m_a)(O1) Y_(l_b, m_b)(O1) Y_(l,-m)(O1) = (-1)^(m_ka) sqrt((2l_a+1)*(2l_b+1))/sqrt(4pi(2l+1)) CG(l_a, l_b, 0, 0, l, 0) CG(l_a, l_b, -m_a, m_b, l, m)
      // int dO2 Y*_ko(O2) Y_ko(O2) Y_lm(O2) = (-1)^(m_ko) int dO2 Y_(l_ko, -m_ko)(O2) Y_(l_ko, m_ko)(O2) Y_(l,m)(O1) = (-1)^(m+m_ko) (2l_ko+1)/sqrt(4pi(2l+1)) CG(l_ko, l_ko, 0, 0, l, 0) CG(l_ko, l_ko, -m_ko, m_ko, l, -m)
      // Term = sum_ko int dr1 int dr2 sum_l=0^inf sum_m=-l^+l [ sqrt((2*l_a+1)*(2l_b+1)) * (2*l_ko+1) / (2*l+1)^2 (-1)^(m+m_a+m_ko) CG(l_a, l_b, 0, 0, l, 0) * CG(l_ko, l_ko, 0, 0, l, 0) * CG(l_a, l_b, -m_a, m_b, l, m) * CG(l_ko, l_ko, -m_ko, m_ko, l, -m) ] * [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_a(r1) psi_b(r1) psi_ko(r2)^2 ]
      // Y^k(r2) = int r_<^k / r_>^(k+1) psi_a(r1) psi_b(r1) r1^2 dr1
      // Term = sum_ko int dr2 sum_l=0^inf sum_m=-l^+l [ sqrt((2*l_a+1)*(2l_b+1)) * (2*l_ko+1) / (2*l+1)^2 (-1)^(m+m_a+m_ko) CG(l_a, l_b, 0, 0, l, 0) * CG(l_ko, l_ko, 0, 0, l, 0) * CG(l_a, l_b, -m_a, m_b, l, m) * CG(l_ko, l_ko, -m_ko, m_ko, l, -m) ] * [ r2^2 Y^l(r2) psi_ko(r2)^2 ]
      if (_o[k1]->spin()*_o[k2]->spin() >= 0) {
        for (int ko = 0; ko < _o.size(); ++ko) {
          lm tlmo(_o[ko]->l(), _o[ko]->m());
          ldouble A = 1;
          if (_o[ko]->spin() == 0) A *= _o[ko]->g();
          for (int ir2 = 0; ir2 < _g->N()-1; ++ir2) {
            ldouble r2 = (*_g)(ir2);
            ldouble r2p1 = (*_g)(ir2+1);
            ldouble dr2 = 0;
            if (ir2 < _g->N()-1) dr2 = (*_g)(ir2+1) - (*_g)(ir2);
    
            for (int l = 0; l <= lmax; ++l) {
              for (int m = -l; m <= l; ++m) {
                ldouble v = std::pow(-1, m+tlm_d1.m+tlmo.m)*(2.0*tlmo.l+1.0)*
                            std::sqrt((2.0*tlm_d1.l+1.0)*(2.0*tlm_d2.l+1.0))*
                            std::pow(2.0*l+1.0, -2)*
                            CG(tlm_d1.l, tlm_d2.l, 0, 0, l, 0)*CG(tlmo.l, tlmo.l, 0, 0, l, 0)*
                            CG(tlm_d1.l, tlm_d2.l, -tlm_d1.m, tlm_d2.m, l, m)*CG(tlmo.l, tlmo.l, -tlmo.m, tlmo.m, l, -m)*
                            _Y[10000*l + 100*k1 + 1*k2][ir2]*
                            A*std::pow(_o[ko]->getNorm(ir2, *_g), 2)*std::pow(r2, 2);
                ldouble vp1 = std::pow(-1, m+tlm_d1.m+tlmo.m)*(2.0*tlmo.l+1.0)*
                            std::sqrt((2.0*tlm_d1.l+1.0)*(2.0*tlm_d2.l+1.0))*
                            std::pow(2.0*l+1.0, -2)*
                            CG(tlm_d1.l, tlm_d2.l, 0, 0, l, 0)*CG(tlmo.l, tlmo.l, 0, 0, l, 0)*
                            CG(tlm_d1.l, tlm_d2.l, -tlm_d1.m, tlm_d2.m, l, m)*CG(tlmo.l, tlmo.l, -tlmo.m, tlmo.m, l, -m)*
                            _Y[10000*l + 100*k1 + 1*k2][ir2+1]*
                            A*std::pow(_o[ko]->getNorm(ir2+1, *_g), 2)*std::pow(r2p1, 2);
                dH(k1, k2) += 0.5*(v+vp1)*dr2;
                _J(k1, k2) += 0.5*(v+vp1)*dr2;
              }
            }
          }
        }
      }

      std::cout << "Calculating matrix element <" << k1 << "|Vex|" << k2 << ">" << std::endl;
      // - full vex
      // vex|u_k1> = sum_ko psi_ko(r1) Y_ko(r1) int psi_ko(r2) psi_k1(r2) Y*_ko(Omega2) Y_k1(Omega2) 1/|r1 - r2| dOmega2 r2^2 dr2
      //
      // dH term will be: <u_k2|vex|u_k1>
      // Term = sum_ko [int dr1 dOmega1 r1^2 psi_k2(r1) psi_ko(r1) Y*_k2(Omega1) Y_ko(Omega1) { int dr2 dOmega2 r2^2 psi_ko(r2) psi_k1(r2) Y*_ko(Omega2) Y_k1(Omega2) 1/|r1 - r2| } ]
      //
      // Term = sum_ko int dr1 int dOmega1 int dr2 int dOmega2 [ r1^2 psi_k2(r1) psi_ko(r1) Y*_k2(Omega1) Y_ko(Omega1) r2^2 psi_ko(r2) psi_k1(r2) Y*_ko(Omega2) Y_k1(Omega2) 1/|r1 - r2| ]
      //
      // 1/|r1 - r2| = sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) r_<^l/r_>^(l+1) Y_lm*(Omega1) Y_lm(Omega2)
      //
      // Term = sum_ko int dr1 int dOmega1 int dr2 int dOmega2 sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_k2(r1) psi_ko(r1) psi_ko(r2) psi_k1(r2) Y*_k2(O1) Y_ko(Omega1) Y*_ko(O2) Y_k1(O2) Y_lm(O1) Y_lm*(O2) ]
      //
      // int dO1 Y*_k2(O1) Y_ko(O1) Y_lm(O1) = (-1)^m_k2 int dO1 Y_(l_k2, -m_k2)(O1) Y_(l_ko, m_ko) Y_lm(O1) = (-1)^(m+m_k2) sqrt( (2l_k2+1) (2l_ko+1))/sqrt(4pi(2l+1)) CG(l_k2, l_ko, 0, 0, l, 0) CG(l_k2, l_ko, -m_k2, m_ko, l, -m)
      //
      // int dO2 Y*_ko(O2) Y_k1(O2) Y_lm*(O2) = (-1)^(m+m_ko) int dO2 Y_(l_ko, -m_ko)(O2) Y_(l_k, m_k)(O2) Y_(l,-m)(O2) = (-1)^(m_ko) sqrt((2l_ko+1) (2l_k1+1))/sqrt(4pi(2l+1)) CG(l_ko, l_k1, 0, 0, l, 0) CG(l_ko, l_k1, -m_ko, m_k1, l, m)
      //
      // Term = sum_ko int dr1 int dr2 sum_l=0^inf sum_m=-l^+l (-1)^(m+m_k2+m_ko) [ (2l_k2+1) * (2l_ko+1) / (2l+1)^2 * CG(l_k2, l_ko, 0, 0, l, 0) * CG(l_k2, l_ko, -m_k2, m_ko, l, -m) * CG(l_ko, l_k1, 0, 0, l, 0) * CG(l_ko, l_k1, -m_ko, m_k1, l, m) ] * [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_k2(r1) psi_ko(r1) psi_ko(r2) psi_k1(r2) ]
      //
      // Y^k(r2) = int r_<^k / r_>^(k+1) psi_a(r1) psi_b(r1) r1^2 dr1
      // Term = sum_ko int dr2 sum_l=0^inf sum_m=-l^+l (-1)^(m+m_k2+m_ko) [ (2l_k2+1) * (2l_ko+1) / (2l+1)^2 * CG(l_k2, l_ko, 0, 0, l, 0) * CG(l_k2, l_ko, -m_k2, m_ko, l, -m) * CG(l_ko, l_k1, 0, 0, l, 0) * CG(l_ko, l_k1, -m_ko, m_k1, l, m) ] * [ r2^2 Y^l(k2,ko,r2) psi_ko(r2) psi_k1(r2) ]
      //
      for (int ko = 0; ko < _o.size(); ++ko) {
        lm tlmo(_o[ko]->l(), _o[ko]->m());
        if (_o[ko]->spin()*_o[k1]->spin() < 0) continue; // spin component dot product
        if (_o[k2]->spin()*_o[ko]->spin() < 0) continue; // spin component dot product
        ldouble A = 1;
        if (_o[ko]->spin() == 0) A *= _o[ko]->g()*0.5;
        for (int ir2 = 0; ir2 < _g->N()-1; ++ir2) {
          ldouble r2 = (*_g)(ir2);
          ldouble r2p1 = (*_g)(ir2+1);
          ldouble dr2 = 0;
          if (ir2 < _g->N()-1) dr2 = (*_g)(ir2+1) - (*_g)(ir2);
    
          for (int l = 0; l <= lmax; ++l) {
            for (int m = -l; m <= l; ++m) {
              ldouble v = std::pow(-1, m + tlm_d2.m + tlmo.m)*(2.0*tlm_d2.l+1.0)*(2.0*tlmo.l+1.0)*std::pow(2.0*l+1.0, -2)*
                          CG(tlm_d2.l, tlmo.l, 0, 0, l, 0)*CG(tlmo.l, tlm_d1.l, 0, 0, l, 0)*
                          CG(tlm_d2.l, tlmo.l, -tlm_d2.m, tlmo.m, l, -m)*CG(tlmo.l, tlm_d1.l, -tlmo.m, tlm_d1.m, l, m)*
                          _Y[10000*l + 100*k2 + 1*ko][ir2]*
                          A*_o[ko]->getNorm(ir2, *_g)*_o[k1]->getNorm(ir2, *_g)*
                          std::pow(r2, 2);
              ldouble vp1 = std::pow(-1, m + tlm_d2.m + tlmo.m)*(2.0*tlm_d2.l+1.0)*(2.0*tlmo.l+1.0)*std::pow(2.0*l+1.0, -2)*
                          CG(tlm_d2.l, tlmo.l, 0, 0, l, 0)*CG(tlmo.l, tlm_d1.l, 0, 0, l, 0)*
                          CG(tlm_d2.l, tlmo.l, -tlm_d2.m, tlmo.m, l, -m)*CG(tlmo.l, tlm_d1.l, -tlmo.m, tlm_d1.m, l, m)*
                          _Y[10000*l + 100*k2 + 1*ko][ir2+1]*
                          A*_o[ko]->getNorm(ir2+1, *_g)*_o[k1]->getNorm(ir2+1, *_g)*
                          std::pow(r2p1, 2);
              dH(k1, k2) += -0.5*(v+vp1)*dr2;
              _K(k1, k2) += 0.5*(v+vp1)*dr2;
            }
          }
        }
      }


    }
  }

  std::cout << "Calculating eigenvalues ..." << std::endl;
  for (int deg_idx = 0; deg_idx < E.size(); ++deg_idx) {
    int deg_order = deg[deg_idx].size();

    MatrixXld dH_deg(deg_order, deg_order);
    dH_deg.setZero();
    MatrixXld S_deg(deg_order, deg_order);
    S_deg.setZero();

    // calculate S = int psi_d1(r) psi_d2(r) Y*_d1(O) Y_d2(O) dr dO = delta_l1l2 delta_m1m2 int psi_d1(r) psi_d2(r) dr
    // and calculate dH = <d1|deltaV|d2>
    for (int d1 = 0; d1 < deg_order; ++d1) {
      int k1 = deg[deg_idx][d1];
      lm tlm_d1(_o[k1]->l(), _o[k1]->m());
      for (int d2 = 0; d2 < deg_order; ++d2) {
        int k2 = deg[deg_idx][d2];
        lm tlm_d2(_o[k2]->l(), _o[k2]->m());
        S_deg(d1, d2) = S(k1, k2);
        dH_deg(d1, d2) = dH(k1, k2);
      }
    }

    for (int ka = 0; ka < deg_order; ++ka) {
      for (int kb = 0; kb < deg_order; ++kb) {
        if (ka == kb) cdeg(idx+ka, idx+kb) = 1.0;
        else cdeg(idx+ka, idx+kb) = 0.0;
      }
    }
    for (int k = 0; k < deg_order; ++k) {
      _Ec[idx+k] = 0;
    }

    MatrixXld SidH = S_deg.inverse()*dH_deg;

    std::cout << "About to calculate eigenvalues for the following states: ";
    std::cout << "States: ";
    for (auto &o : deg[deg_idx]) std::cout << " " << o;
    std::cout << std::endl;
    std::cout << "dH: " << std::endl;
    std::cout << dH << std::endl;
    std::cout << "S: " << std::endl;
    std::cout << S << std::endl;
    std::cout << "S^-1 H: " << std::endl;
    std::cout << SidH << std::endl;
    std::cout << "det(S^-1 H): " << std::endl;
    std::cout << SidH.determinant() << std::endl;

    EigenSolver<MatrixXld> solver(SidH);
    // translate indices to idx, idx+1, etc.
    if (solver.info() == Eigen::Success) {
      std::cout << "Found eigenvalues for idx = " << idx << " "<< solver.eigenvalues() << std::endl;
      std::cout << "Eigenvectors = " << solver.eigenvectors() << std::endl;
      for (int k = 0; k < deg_order; ++k) {
        _Ec[idx+k] = solver.eigenvalues()(k).real();
      }
      for (int ka = 0; ka < deg_order; ++ka) {
        for (int kb = 0; kb < deg_order; ++kb) {
          cdeg(idx+ka, idx+kb) = solver.eigenvectors()(ka, kb);
        }
      }
    } else {
      std::cout << "Failed to calculate eigenvectors for this set of degenerate states. Assuming no mixing." << std::endl;
      std::cout << "States: ";
      for (auto &o : deg[deg_idx]) std::cout << " " << o;
      std::cout << std::endl;
      std::cout << "Eigenvalues = ";
      for (int k = 0; k < deg_order; ++k) std::cout << " " << _Ec[idx+k];
      std::cout << std::endl;
      std::cout << "dH: " << std::endl;
      std::cout << dH << std::endl;
      std::cout << "S: " << std::endl;
      std::cout << S << std::endl;
      std::cout << "S^-1 H: " << std::endl;
      std::cout << SidH << std::endl;
      std::cout << "det(S^-1 H): " << std::endl;
      std::cout << SidH.determinant() << std::endl;
    }
    idx += deg_order;
  }

  for (int ka = 0; ka < E.size(); ++ka) {
    int deg_order_a = deg[ka].size();
    for (int kad = 0; kad < deg_order_a; ++kad) {
      for (int kb = 0; kb < E.size(); ++kb) {
        if (kb == ka) continue;
        int deg_order_b = deg[kb].size();
        for (int kbd = 0; kbd < deg_order_b; ++kbd) {
          _c(deg[ka][kad], deg[kb][kbd]) += dH(deg[ka][kad], deg[kb][kbd])/(E[kb] - E[ka]);
        }
      }
    }
  }
  _c = _c*cdeg;
  for (int i = 0; i < _o.size(); ++i) {
    _c(i, i) += 1.0;
  }
  std::cout << "Coefficients:" << std::endl;
  std::cout << _c << std::endl;

  std::cout << "J:" << std::endl;
  std::cout << _J << std::endl;

  std::cout << "K:" << std::endl;
  std::cout << _K << std::endl;


  std::cout << "Recalculating J and K after correction" << std::endl;

  // now recalculate J and K with corrected orbitals
  for (int k1 = 0; k1 < _o.size(); ++k1) { // for each orbital k1
    lm tlm_d1(_o[k1]->l(), _o[k1]->m());
    for (int k2 = 0; k2 < _o.size(); ++k2) { // for each orbital k2
      lm tlm_d2(_o[k2]->l(), _o[k2]->m());

      std::cout << "Calculating matrix element <" << k1 << "|Vd|" << k2 << ">" << std::endl;
      if (_o[k1]->spin()*_o[k2]->spin() < 0) continue; // only if the same spin

      for (int ko = 0; ko < _o.size(); ++ko) { // project on orbitals ko
        lm tlmo(_o[ko]->l(), _o[ko]->m());
        for (int ir2 = 0; ir2 < _g->N()-1; ++ir2) { // integrate on r1
          ldouble r2 = (*_g)(ir2);
          ldouble r2p1 = (*_g)(ir2+1);
          ldouble dr2 = 0;
          if (ir2 < _g->N()-1) dr2 = (*_g)(ir2+1) - (*_g)(ir2);
    
          // sum on l and m due to 1/r12 = sum_l sum_m 4 pi/(2l + 1) int Ylm r_<^l/r_>^(l+1) dr
          // the integral is absorbed in the Y factor below
          for (int l = 0; l <= lmax; ++l) {
            for (int m = -l; m <= l; ++m) {

              // the new orbital k1 and k2 are expressed as a linear combination of the original k1 and k2
              // so:
              // o^new_k1 = sum_k1c c(k1, k1c) o_k1c
              // o^new_k2 = sum_k2c c(k2, k2c) o_k2c
              ldouble v = 0;
              for (int k1c = 0; k1c < _o.size(); ++k1c) {
                lm tlm_d1c(_o[k1c]->l(), _o[k1c]->m());
                ldouble ck1 = _c(k1, k1c).real();
                if (ck1 == 0) continue;

                for (int k2c = 0; k2c < _o.size(); ++k2c) {
                  lm tlm_d2c(_o[k2c]->l(), _o[k2c]->m());
                  ldouble ck2 = _c(k2, k2c).real();
                  if (ck2 == 0) continue;

                  for (int koc = 0; koc < _o.size(); ++koc) {
                    lm tlm_doc(_o[koc]->l(), _o[koc]->m());
                    ldouble oko = _c(ko, koc).real()*_o[koc]->getNorm(ir2, *_g);
                    ldouble okop1 = _c(ko, koc).real()*_o[koc]->getNorm(ir2+1, *_g);
                    if (oko == 0 && okop1 == 0) continue;

                    ldouble A = 1;
                    if (_o[ko]->spin() == 0) A *= _o[ko]->g();

                    ldouble fn = std::pow(-1, m+tlm_d1c.m+tlm_doc.m)*(2.0*tlm_doc.l+1.0)*
                         std::sqrt((2.0*tlm_d1c.l+1.0)*(2.0*tlm_d2c.l+1.0))*std::pow(2.0*l+1.0, -2)*
                         CG(tlm_d1c.l, tlm_d2c.l, 0, 0, l, 0)*CG(tlm_doc.l, tlm_doc.l, 0, 0, l, 0)*
                         CG(tlm_d1c.l, tlm_d2c.l, -tlm_d1c.m, tlm_d2c.m, l, m)*CG(tlm_doc.l, tlm_doc.l, -tlm_doc.m, tlm_doc.m, l, -m)*
                         _Y[10000*l + 100*k1c + 1*k2c][ir2]*ck1*ck2*
                         std::pow(oko, 2)*
                         A*std::pow(r2, 2)*dr2;
                    ldouble fnp1 = std::pow(-1, m+tlm_d1c.m+tlm_doc.m)*(2.0*tlm_doc.l+1.0)*
                         std::sqrt((2.0*tlm_d1c.l+1.0)*(2.0*tlm_d2c.l+1.0))*std::pow(2.0*l+1.0, -2)*
                         CG(tlm_d1c.l, tlm_d2c.l, 0, 0, l, 0)*CG(tlm_doc.l, tlm_doc.l, 0, 0, l, 0)*
                         CG(tlm_d1c.l, tlm_d2c.l, -tlm_d1c.m, tlm_d2c.m, l, m)*CG(tlm_doc.l, tlm_doc.l, -tlm_doc.m, tlm_doc.m, l, -m)*
                         _Y[10000*l + 100*k1c + 1*k2c][ir2+1]*ck1*ck2*
                         std::pow(okop1, 2)*
                         A*std::pow(r2p1, 2);
                    v += 0.5*(fn+fnp1)*dr2;
                  }
                }
              }

              _Jcorr(k1, k2) += v;
            } // for m in sum_m
          } // for l in sum_l
        } // integrate on r2
      } // for each ko, on which to project k1 and k2
    } // for each k2
  } // for each k1
      
  for (int k1 = 0; k1 < _o.size(); ++k1) { // for each orbital k1
    lm tlm_d1(_o[k1]->l(), _o[k1]->m());
    for (int k2 = 0; k2 < _o.size(); ++k2) { // for each orbital k2
      lm tlm_d2(_o[k2]->l(), _o[k2]->m());

      std::cout << "Calculating matrix element <" << k1 << "|Vex|" << k2 << ">" << std::endl;
      for (int ko = 0; ko < _o.size(); ++ko) { // project on orbitals ko
        lm tlmo(_o[ko]->l(), _o[ko]->m());
        if (_o[ko]->spin()*_o[k1]->spin() < 0) continue; // spin component dot product
        if (_o[k2]->spin()*_o[ko]->spin() < 0) continue; // spin component dot product
        // integrate on r2
        for (int ir2 = 0; ir2 < _g->N()-1; ++ir2) {
          ldouble r2 = (*_g)(ir2);
          ldouble r2p1 = (*_g)(ir2+1);
          ldouble dr2 = 0;
          if (ir2 < _g->N()-1) dr2 = (*_g)(ir2+1) - (*_g)(ir2);
    
          // sum on l and m due to 1/r12 = sum_l sum_m 4 pi/(2l + 1) int Ylm r_<^l/r_>^(l+1) dr
          // the integral is absorbed in the Y factor below
          for (int l = 0; l <= lmax; ++l) {
            for (int m = -l; m <= l; ++m) {

              // the new orbital k1 and k2 are expressed as a linear combination of the original k1 and k2
              // so:
              // o^new_k1 = sum_k1c c(k1, k1c) o_k1c
              // o^new_k2 = sum_k2c c(k2, k2c) o_k2c
              ldouble v = 0;
              for (int k1c = 0; k1c < _o.size(); ++k1c) {
                lm tlm_d1c(_o[k1c]->l(), _o[k1c]->m());
                ldouble ok1 = _c(k1, k1c).real()*_o[k1c]->getNorm(ir2, *_g);
                ldouble ok1p1 = _c(k1, k1c).real()*_o[k1c]->getNorm(ir2+1, *_g);
                if (ok1 == 0) continue;

                for (int k2c = 0; k2c < _o.size(); ++k2c) {
                  lm tlm_d2c(_o[k2c]->l(), _o[k2c]->m());
                  ldouble ck2 = _c(k2, k2c).real();
                  if (ck2 == 0) continue;

                  for (int koc = 0; koc < _o.size(); ++koc) {
                    lm tlm_doc(_o[koc]->l(), _o[koc]->m());
                    ldouble ckor1 = _c(ko, koc).real();
                    ldouble okor2 = _c(ko, koc).real()*_o[koc]->getNorm(ir2, *_g);
                    ldouble okor2p1 = _c(ko, koc).real()*_o[koc]->getNorm(ir2+1, *_g);
                    if (ckor1 == 0 || okor2 == 0) continue;

                    ldouble A = 1;
                    if (_o[ko]->spin() == 0) A *= _o[ko]->g()*0.5;

                    ldouble fn = std::pow(-1, m + tlm_d2c.m + tlm_doc.m)*
                         (2.0*tlm_d2c.l+1.0)*(2.0*tlm_doc.l+1.0)*
                         std::pow(2.0*l+1.0, -2)*
                         CG(tlm_d2c.l, tlm_doc.l, 0, 0, l, 0)*CG(tlm_doc.l, tlm_d1c.l, 0, 0, l, 0)*
                         CG(tlm_d2c.l, tlm_doc.l, -tlm_d2c.m, tlm_doc.m, l, -m)*CG(tlm_doc.l, tlm_d1c.l, -tlm_doc.m, tlm_d1c.m, l, m)*
                         ck2*ckor1*_Y[10000*l + 100*k2c + 1*koc][ir2]*
                         okor2*ok1*A*std::pow(r2, 2);
                    ldouble fnp1 = std::pow(-1, m + tlm_d2c.m + tlm_doc.m)*
                         (2.0*tlm_d2c.l+1.0)*(2.0*tlm_doc.l+1.0)*
                         std::pow(2.0*l+1.0, -2)*
                         CG(tlm_d2c.l, tlm_doc.l, 0, 0, l, 0)*CG(tlm_doc.l, tlm_d1c.l, 0, 0, l, 0)*
                         CG(tlm_d2c.l, tlm_doc.l, -tlm_d2c.m, tlm_doc.m, l, -m)*CG(tlm_doc.l, tlm_d1c.l, -tlm_doc.m, tlm_d1c.m, l, m)*
                         ck2*ckor1*_Y[10000*l + 100*k2c + 1*koc][ir2+1]*
                         okor2p1*ok1p1*A*std::pow(r2p1, 2);
                    v += 0.5*(fn+fnp1)*dr2;
                  }
                }
              }
              _Kcorr(k1, k2) += v;
            }
          }
        }
        
      }


    }
  }

  std::cout << "Corrected E0:" << std::endl;
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    ldouble A = 1;
    if (_o[k]->spin() == 0) A *= _o[k]->g();
    E0 += A*(_Ec[k] + _o[k]->E());
  }
  for (int i = 0; i < _o.size(); ++i) {
    for (int j = 0; j < _o.size(); ++j) {
      ldouble A = 1;
      if (_o[j]->spin() == 0) A *= _o[j]->g();
      //if (_o[i]->spin() == 0) A *= _o[i]->g();
      E0 += -0.5*(A*_Jcorr(i, j) - A*_Kcorr(i, j));
    }
  }
  std::cout << E0 << std::endl;
}

ldouble NonCentralCorrection::getE0() {
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    ldouble A = 1;
    if (_o[k]->spin() == 0) A *= _o[k]->g();
    E0 += A*(_Ec[k] + _o[k]->E());
  }
  for (int i = 0; i < _o.size(); ++i) {
    for (int j = 0; j < _o.size(); ++j) {
      ldouble A = 1;
      if (_o[j]->spin() == 0) A *= _o[j]->g();
      //if (_o[i]->spin() == 0) A *= _o[i]->g();
      E0 += -0.5*(A*_Jcorr(i, j) - A*_Kcorr(i, j));
    }
  }
  return E0;
}

ldouble NonCentralCorrection::getE0Uncorrected() {
  ldouble E0 = 0;
  for (int k = 0; k < _o.size(); ++k) {
    ldouble A = 1;
    if (_o[k]->spin() == 0) A *= _o[k]->g();
    E0 += A*_o[k]->E();
  }
  ldouble J = 0;
  ldouble K = 0;
  for (auto &vditm : _vd) {
    int k = vditm.first;
    int l = _o[k]->l();
    int m = _o[k]->m();
    ldouble A = 1;
    if (_o[k]->spin() == 0) A *= _o[k]->g();
    for (int ir = 0; ir < _g->N()-1; ++ir) {
      ldouble r = (*_g)(ir);
      ldouble rp1 = (*_g)(ir+1);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      ldouble fn = A*_vd[k][ir]*std::pow(_o[k]->getNorm(ir, *_g), 2)*std::pow(r, 2);
      ldouble fnp1 = A*_vd[k][ir+1]*std::pow(_o[k]->getNorm(ir+1, *_g), 2)*std::pow(rp1, 2);
      J += 0.5*(fn+fnp1)*dr;
    }
  }
  for (auto &vexitm : _vex) {
    const int k1 = vexitm.first.first;
    const int k2 = vexitm.first.second;
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();
    int l2 = _o[k2]->l();
    int m2 = _o[k2]->m();
    ldouble A = 1;
    if (_o[k2]->spin() == 0) A *= _o[k2]->g();
    for (int ir = 0; ir < _g->N()-1; ++ir) {
      ldouble r = (*_g)(ir);
      ldouble rp1 = (*_g)(ir+1);
      ldouble dr = 0;
      if (ir < _g->N()-1)
        dr = (*_g)(ir+1) - (*_g)(ir);
      ldouble fn = A*_vex[std::pair<int,int>(k1, k2)][ir]*_o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2);
      ldouble fnp1 = A*_vex[std::pair<int,int>(k1, k2)][ir+1]*_o[k1]->getNorm(ir+1, *_g)*_o[k2]->getNorm(ir+1, *_g)*std::pow(rp1, 2);
      K += 0.5*(fn+fnp1)*dr;
    }
  }
  E0 += -0.5*(J - K);
  return E0;
}


