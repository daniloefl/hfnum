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
  _Ec.resize(_o.size(), 0);
  for (auto &e : _Ec) e = 0;

  _c.resize(_o.size(), _o.size());
  _c.setZero();

  MatrixXcld cdeg(_o.size(), _o.size());
  cdeg.setZero();

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
    lm tlm_d1(_o[k1]->initialL(), _o[k1]->initialM());
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      lm tlm_d2(_o[k2]->initialL(), _o[k2]->initialM());
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          ldouble dr = 0;
          if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
          S(k1, k2) += _o[k1]->getNorm(ir, tlm_d1.l, tlm_d1.m, *_g)*_o[k2]->getNorm(ir, tlm_d2.l, tlm_d2.m, *_g)*std::pow(r, 2)*dr;
        }
      }

      // now calculate dH, where the effect of all degenerate states is considered
      // the current inaccurate result follows
      // - vd
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          ldouble dr = 0;
          if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
          dH(k1, k2) += -_vd[k1][ir]*_o[k1]->getNorm(ir, tlm_d1.l, tlm_d1.m, *_g)*_o[k2]->getNorm(ir, tlm_d2.l, tlm_d2.m, *_g)*std::pow(r, 2)*dr;
        }
      }

      // + vex
      for (int ko = 0; ko < _o.size(); ++ko) {
        lm tlmo(_o[ko]->initialL(), _o[ko]->initialM());
        if (tlmo.l != tlm_d2.l || tlmo.m != tlm_d2.m || _o[ko]->spin()*_o[k2]->spin() < 0) continue;
        if (tlmo.l != tlm_d1.l || tlmo.m != tlm_d1.m || _o[ko]->spin()*_o[k1]->spin() < 0) continue;
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          ldouble dr = 0;
          if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
          dH(k1, k2) += _vex[std::pair<int, int>(k1, ko)][ir]*_o[ko]->getNorm(ir, tlmo.l, tlmo.m, *_g)*_o[k2]->getNorm(ir, tlm_d2.l, tlm_d2.m, *_g)*std::pow(r, 2)*dr;
        }
      }


      // + full vd
      // vd(r1) = sum_ko int |psi_ko(r2) Y_ko(Omega2)|^2 1/|r1 - r2| dOmega2 r2^2 dr2
      // dH(a,b) term will be: Term = sum_ko [int dr1 dOmega1 r1^2 psi_a(r1) psi_b(r1) Y*_a(O1) Y_b(O1) { int dr2 dOmega2 r2^2 psi_ko(r2)^2 Y_ko(Omega2)^2 1/|r1 - r2| } ]
      // Term = sum_ko int dr1 int dOmega1 int dr2 int dOmega2 [ r1^2 psi_a(r1) psi_b(r1) Y*_a(O1) Y_b(O1) r2^2 psi_ko(r2)^2 Y_ko(Omega2)^2 1/|r1 - r2| ]
      // 1/|r1 - r2| = sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) r_<^l/r_>^(l+1) Y_lm*(Omega1) Y_lm(Omega2)
      // Term = sum_ko int dr1 int dOmega1 int dr2 int dOmega2 sum_l=0^inf sum_m=-l^+l 4 pi/(2*l+1) [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_a(r1) psi_b(r1) psi_ko(r2)^2 Y*_a(O1) Y_b(O1) Y_ko(O2)^2 Y_lm(O1) Y_lm*(O2) ]
      // int dO1 Y*_a(O1) Y_b(O1) Y_lm*(O1) = (-1)^(m+m_a) int dO1 Y_(l_a, -m_a)(O1) Y_(l_b, m_b)(O1) Y_(l,-m)(O1) = (-1)^(m_ka sqrt((2l_a+1)*(2l_b+1))/sqrt(4pi(2l+1)) CG(l_a, l_b, 0, 0, l, 0) CG(l_a, l_b, -m_a, m_b, l, m)
      // int dO2 Y*_ko(O2) Y_ko(O2) Y_lm(O2) = (-1)^(m_ko) int dO2 Y_(l_ko, -m_ko)(O2) Y_(l_ko, m_ko)(O2) Y_(l,m)(O1) = (-1)^(m+m_ko) (2l_ko+1)/sqrt(4pi(2l+1)) CG(l_ko, l_ko, 0, 0, l, 0) CG(l_ko, l_ko, -m_ko, m_ko, l, -m)
      // Term = sum_ko int dr1 int dr2 sum_l=0^inf sum_m=-l^+l [ sqrt((2*l_a+1)*(2l_b+1)) * (2*l_ko+1) / (2*l+1)^2 (-1)^(m+m_a+m_ko) CG(l_a, l_b, 0, 0, l, 0) * CG(l_ko, l_ko, 0, 0, l, 0) * CG(l_a, l_b, -m_a, m_b, l, m) * CG(l_ko, l_ko, -m_ko, m_ko, l, -m) ] * [ r1^2 r2^2 r_<^l/r_>^(l+1) psi_a(r1) psi_b(r1) psi_ko(r2)^2 ]
      //
      if (_o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ko = 0; ko < _o.size(); ++ko) {
          lm tlmo(_o[ko]->initialL(), _o[ko]->initialM());
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
                  dH(k1, k2) += std::pow(-1, m+tlm_d1.m+tlmo.m)*(2.0*tlmo.l+1.0)*std::sqrt((2.0*tlm_d1.l+1.0)*(2.0*tlm_d2.l+1.0))/std::pow(2.0*l+1.0, 2)*CG(tlm_d1.l, tlm_d2.l, 0, 0, l, 0)*CG(tlmo.l, tlmo.l, 0, 0, l, 0)*CG(tlm_d1.l, tlm_d2.l, -tlm_d1.m, tlm_d2.m, l, m)*CG(tlmo.l, tlmo.l, -tlmo.m, tlmo.m, l, -m)*_o[k1]->getNorm(ir1, tlm_d1.l, tlm_d1.m, *_g)*_o[k2]->getNorm(ir1, tlm_d2.l, tlm_d2.m, *_g)*std::pow(_o[ko]->getNorm(ir2, tlmo.l, tlmo.m, *_g), 2)*std::pow(r1*r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr1*dr2;
                }
              }
            }
          }
        }
      }

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
      for (int ko = 0; ko < _o.size(); ++ko) {
        lm tlmo(_o[ko]->initialL(), _o[ko]->initialM());
        if (_o[ko]->spin()*_o[k1]->spin() < 0) continue; // spin component dot product
        if (_o[k2]->spin()*_o[ko]->spin() < 0) continue; // spin component dot product
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
                dH(k1, k2) += -std::pow(-1, m + tlm_d2.m + tlmo.m)*(2.0*tlm_d2.l+1.0)*(2.0*tlmo.l+1.0)/std::pow(2.0*l+1.0, 2)*CG(tlm_d2.l, tlmo.l, 0, 0, l, 0)*CG(tlmo.l, tlm_d1.l, 0, 0, l, 0)*CG(tlm_d2.l, tlmo.l, -tlm_d2.m, tlmo.m, l, -m)*CG(tlmo.l, tlm_d1.l, -tlmo.m, tlm_d1.m, l, m)*_o[k2]->getNorm(ir1, tlm_d2.l, tlm_d2.m, *_g)*_o[ko]->getNorm(ir1, tlmo.l, tlmo.m, *_g)*_o[ko]->getNorm(ir2, tlmo.l, tlmo.m, *_g)*_o[k1]->getNorm(ir2, tlm_d1.l, tlm_d1.m, *_g)*std::pow(r1*r2, 2)*std::pow(rsmall, l)/std::pow(rlarge, l+1)*dr1*dr2;
              }
            }
          }
        }
      }


    }
  }

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
      lm tlm_d1(_o[k1]->initialL(), _o[k1]->initialM());
      for (int d2 = 0; d2 < deg_order; ++d2) {
        int k2 = deg[deg_idx][d2];
        lm tlm_d2(_o[k2]->initialL(), _o[k2]->initialM());
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
  //std::cout << _c << std::endl;
}

