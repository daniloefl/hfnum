#include "NonCentralCorrection.h"
#include "StateReader.h"
#include "Orbital.h"
#include "Grid.h"
#include <vector>
#include <map>
#include <string>
#include <cmath>

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
  _g->reset((bool) sr.getInt("grid.isLog"), sr.getDouble("grid.dx"), sr.getInt("grid.N"), sr.getDouble("grid.rmin"));
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

void NonCentralCorrection::correct() {
  int lmax = 2; // approximated here
  // define delta V = (full Vd - full Vex) - (Vd - Vex)
  // this is defined separately for each orbital equation: we want the error in the eigenenergies
  _Ec.resize(_o.size(), 0);
  for (int k = 0; k < _o.size(); ++k) {
    lm tlm(_o[k]->initialL(), _o[k]->initialM());

    // the current inaccurate result follows
    // - vd
    for (int ir = 0; ir < _g->N(); ++ir) {
      ldouble r = (*_g)(ir);
      ldouble dr = 0;
      if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
      _Ec[k] += -_vd[k][ir]*std::pow(_o[k]->getNorm(ir, tlm.l, tlm.m, *_g), 2)*std::pow(r, 2)*dr;
    }

    // + full vd
    // vd(r1) = sum_k2 int |psi_k2(r2) Y_k2(Omega2)|^2 1/|r1 - r2| dOmega2 r2^2 dr2
    // E1 term will be: Term = sum_k2 [int dr1 dOmega1 r1^2 psi_k(r1)^2 Y*_k(Omega1)^2 { int dr2 dOmega2 r2^2 psi_k2(r2)^2 Y_k2(Omega2)^2 1/|r1 - r2| } ]
    // Term = sum_k2 int dr1 int dOmega1 int dr2 int dOmega2 [ r1^2 psi_k(r1)^2 Y*_k(Omega1)^2 r2^2 psi_k2(r2)^2 Y_k2(Omega2)^2 1/|r1 - r2| ]
    // 1/|r1 - r2| = sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) r_<^l/r_>^(l+1) Y_lm*(Omega1) Y_lm(Omega2)
    // Term = sum_k2 int dr1 int dOmega1 int dr2 int dOmega2 sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_k(r1)^2 psi_k2(r2)^2 Y*_k(O1)^2 Y_k2(O2)^2 Y_lm(O1) Y_lm*(O2) ]
    // int dO1 Y*_k(O1) Y_k(O1) Y_lm*(O1) = (-1)^(m+m_k) int dO1 Y_(l_k, -m_k)(O1) Y_(l_k, m_k)(O1) Y_(l,-m)(O1) = (-1)^(m_k) (2l_k+1)/sqrt(4pi(2l+1)) CG(l_k, l_k, 0, 0, l, 0) CG(l_k, l_k, -m_k, m_k, l, m)
    // int dO2 Y*_k2(O2) Y_k2(O2) Y_lm(O2) = (-1)^(m_k2) int dO2 Y_(l_k2, -m_k2)(O2) Y_(l_k2, m_k2)(O2) Y_(l,m)(O1) = (-1)^(m+m_k2) (2l_k2+1)/sqrt(4pi(2l+1)) CG(l_k2, l_k2, 0, 0, l, 0) CG(l_k2, l_k2, -m_k2, m_k2, l, -m)
    // Term = sum_k2 int dr1 int dr2 sum_l=0^inf sum_m=-l^+l [ (2*l_k+1) * (2*l_k2+1) / (2*l+1)^2 (-1)^(m+m_k+m_k2) CG(l_k, l_k, 0, 0, l, 0) * CG(l_k2, l_k2, 0, 0, l, 0) * CG(l_k, l_k, -m_k, m_k, l, m) * CG(l_k2, l_k2, -m_k2, m_k2, l, -m) ] * [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_k(r1)^2 psi_k2(r2)^2 ]
    //
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      lm tlm2(_o[k2]->initialL(), _o[k2]->initialM());
      for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
        ldouble r1 = (*_g)(ir1);
        ldouble dr1 = 0;
        if (ir1 < _g->N()-1) dr1 = (*_g)(ir1+1) - (*_g)(ir1);

        for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
          ldouble r2 = (*_g)(ir2);
          ldouble dr2 = 0;
          if (ir2 < _g->N()-1) dr2 = (*_g)(ir2+1) - (*_g)(ir2);

          ldouble rsmall = r1;
          ldouble rlarge = r2;
          if (r2 < r1) {
            rsmall = r2;
            rlarge = r1;
          }

          for (int l = 0; l <= lmax; ++l) {
            for (int m = -l; m <= l; ++m) {
              _Ec[k] += std::pow(-1, m+tlm.m+tlm2.m)*(2.0*tlm2.l+1.0)*(2.0*tlm.l+1.0)/std::pow(2.0*l+1.0, 2)*CG(tlm.l, tlm.l, 0, 0, l, 0)*CG(tlm2.l, tlm2.l, 0, 0, l, 0)*CG(tlm.l, tlm.l, -tlm.m, tlm.m, l, m)*CG(tlm2.l, tlm2.l, -tlm2.m, tlm2.m, l, -m)*std::pow(_o[k]->getNorm(ir1, tlm.l, tlm.m, *_g)*_o[k2]->getNorm(ir2, tlm2.l, tlm2.m, *_g), 2)*std::pow(r1*r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr1*dr2;
            }
          }
        }
      }
    }

    // + vex
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      lm tlm2(_o[k2]->initialL(), _o[k2]->initialM());
      for (int ir = 0; ir < _g->N(); ++ir) {
        ldouble r = (*_g)(ir);
        ldouble dr = 0;
        if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
        _Ec[k] += _vex[std::pair<int, int>(k, k2)][ir]*_o[k]->getNorm(ir, tlm.l, tlm.m, *_g)*_o[k2]->getNorm(ir, tlm2.l, tlm2.m, *_g)*std::pow(r, 2)*dr;
      }
    }


    // - full vex
    // vex(r1) psi_k2(r1) Y_k2(r1) = sum_k2 psi_k2(r1) Y_k2(r1) int psi_k2(r2) psi_k(r2) Y*_k2(Omega2) Y_k(Omega2) 1/|r1 - r2| dOmega2 r2^2 dr2
    //
    // E1 term will be: Term = sum_k2 [int dr1 dOmega1 r1^2 psi_k2(r1) psi_k(r1) Y_k2(Omega1) Y*_k(Omega1) { int dr2 dOmega2 r2^2 psi_k2(r2) psi_k(r2) Y*_k2(Omega2) Y_k(Omega2) 1/|r1 - r2| } ]
    //
    // Term = sum_k2 int dr1 int dOmega1 int dr2 int dOmega2 [ r1^2 psi_k2(r1) psi_k(r1) Y_k2(Omega1) Y*_k(Omega1) r2^2 psi_k2(r2) psi_k(r2) Y*_k2(Omega2) Y_k(Omega2) 1/|r1 - r2| ]
    //
    // 1/|r1 - r2| = sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) r_<^l/r_>^(l+1) Y_lm*(Omega1) Y_lm(Omega2)
    //
    // Term = sum_k2 int dr1 int dOmega1 int dr2 int dOmega2 sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_k(r1) psi_k2(r1) psi_k2(r2) psi_k(r2) Y*_k(O1) Y_k2(Omega1) Y*_k2(O2) Y_k(O2) Y_lm(O1) Y_lm*(O2) ]
    //
    // int dO1 Y*_k(O1) Y_k2(O1) Y_lm(O1) = (-1)^m_k int dO1 Y_(l_k, -m_k)(O1) Y_(l_k2, m_k2) Y_lm(O1) = (-1)^(m+m_k) sqrt( (2l_k+1) (2l_k2+1))/sqrt(4pi(2l+1)) CG(l_k, l_k2, 0, 0, l, 0) CG(l_k, l_k2, -m_k, m_k2, l, -m)
    //
    // int dO2 Y*_k2(O2) Y_k(O2) Y_lm*(O2) = (-1)^(m+m_k2) int dO2 Y_(l_k2, -m_k2)(O2) Y_(l_k, m_k)(O2) Y_(l,-m)(O2) = (-1)^(m_k2) sqrt((2l_k2+1) (2l_k+1))/sqrt(4pi(2l+1)) CG(l_k2, l_k, 0, 0, l, 0) CG(l_k2, l_k, -m_k2, m_k, l, m)
    //
    // Term = sum_k2 int dr1 int dr2 sum_l=0^inf sum_m=-l^+l (-1)^(m+m_k+m_k2) [ (2l_k+1) * (2l_k2+1) / (2l+1)^2 * CG(l_k, l_k2, 0, 0, l, 0) * CG(l_k, l_k2, -m_k, m_k2, l, -m) * CG(l_k2, l_k, 0, 0, l, 0) * CG(l_k2, l_k, -m_k2, m_k, l, m) ] * [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_k(r1) psi_k2(r1) psi_k2(r2) psi_k(r2) ]
    //
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      lm tlm2(_o[k2]->initialL(), _o[k2]->initialM());
      if (_o[k2]->spin()*_o[k]->spin() < 0) continue; // spin component dot product
      for (int ir1 = 0; ir1 < _g->N(); ++ir1) {
        ldouble r1 = (*_g)(ir1);
        ldouble dr1 = 0;
        if (ir1 < _g->N()-1) dr1 = (*_g)(ir1+1) - (*_g)(ir1);

        for (int ir2 = 0; ir2 < _g->N(); ++ir2) {
          ldouble r2 = (*_g)(ir2);
          ldouble dr2 = 0;
          if (ir2 < _g->N()-1) dr2 = (*_g)(ir2+1) - (*_g)(ir2);

          ldouble rsmall = r1;
          ldouble rlarge = r2;
          if (r2 < r1) {
            rsmall = r2;
            rlarge = r1;
          }

          for (int l = 0; l <= lmax; ++l) {
            for (int m = -l; m <= l; ++m) {
              _Ec[k] += -std::pow(-1, m + tlm.m + tlm2.m)*(2.0*tlm2.l+1.0)*(2.0*tlm.l+1.0)/std::pow(2.0*l+1.0, 2)*CG(tlm.l, tlm2.l, 0, 0, l, 0)*CG(tlm2.l, tlm.l, 0, 0, l, 0)*CG(tlm.l, tlm2.l, -tlm.m, tlm2.m, l, -m)*CG(tlm2.l, tlm.l, -tlm2.m, tlm.m, l, m)*_o[k]->getNorm(ir1, tlm.l, tlm.m, *_g)*_o[k]->getNorm(ir2, tlm.l, tlm.m, *_g)*_o[k2]->getNorm(ir2, tlm2.l, tlm2.m, *_g)*_o[k2]->getNorm(ir1, tlm2.l, tlm2.m, *_g)*std::pow(r1*r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr1*dr2;
            }
          }
        }
      }
    }
  }
}

