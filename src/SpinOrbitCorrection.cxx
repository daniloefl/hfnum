#include "SpinOrbitCorrection.h"
#include "StateReader.h"
#include "Orbital.h"
#include "Grid.h"
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <iostream>

#include <Eigen/Eigenvalues> 

SpinOrbitCorrection::SpinOrbitCorrection() {
}

SpinOrbitCorrection::~SpinOrbitCorrection() {
}

void SpinOrbitCorrection::load(const std::string &fname) {
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

void SpinOrbitCorrection::correct() {
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

  // define delta V = (full Vd - full Vex) - (Vd - Vex)
  // this is defined separately for each orbital equation: we want the error in the eigenenergies
  for (int k1 = 0; k1 < _o.size(); ++k1) {
    lm tlm_d1(_o[k1]->l(), _o[k1]->m());
    for (int k2 = 0; k2 < _o.size(); ++k2) {
      lm tlm_d2(_o[k2]->l(), _o[k2]->m());
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ir = 0; ir < _g->N(); ++ir) {
          ldouble r = (*_g)(ir);
          ldouble dr = 0;
          if (ir < _g->N()-1) dr = (*_g)(ir+1) - (*_g)(ir);
          S(k1, k2) += _o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 2)*dr;
        }
      }

      // now calculate dH
      // 1/(2r) dV/dr L . S |u1> = 1/(4r) dV/dr (J^2 - L^2 - S^2) |u1>
      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble dvdr = _Z*std::pow(r, -2);
          ldouble s1 = 0.5;
          ldouble l1 = tlm_d1.l;
          ldouble j1 = l1 + s1;
          ldouble coeff = j1*(j1+1) - l1*(l1+1) - s1*(s1+1);
          dH(k1, k2) += 0.25*std::pow(137.035999139, -2)*dvdr*coeff*_o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 1)*dr;
        }
      }

      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ir = 0; ir < _g->N()-1; ++ir) {
          ldouble r = (*_g)(ir);
          ldouble dr = (*_g)(ir+1) - (*_g)(ir);
          ldouble dvdr = (_vd[k1][ir+1] - _vd[k1][ir])/dr;
          ldouble s1 = 0.5;
          ldouble l1 = tlm_d1.l;
          ldouble j1 = l1 + s1;
          ldouble coeff = j1*(j1+1) - l1*(l1+1) - s1*(s1+1);
          dH(k1, k2) += 0.25*std::pow(137.035999139, -2)*dvdr*coeff*_o[k1]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 1)*dr;
        }
      }

      if (tlm_d1.l == tlm_d2.l && tlm_d1.m == tlm_d2.m && _o[k1]->spin()*_o[k2]->spin() > 0) {
        for (int ko = 0; ko < _o.size(); ++ko) {
          lm tlmo(_o[ko]->l(), _o[ko]->m());
          if (_o[ko]->spin()*_o[k1]->spin() < 0) continue; // vex == 0 here
          for (int ir = 0; ir < _g->N()-1; ++ir) {
            ldouble r = (*_g)(ir);
            ldouble dr = (*_g)(ir+1) - (*_g)(ir);
            ldouble dvdr = -(_vex[std::pair<int,int>(k1, ko)][ir+1] - _vex[std::pair<int,int>(k1, ko)][ir])/dr;
            ldouble s1 = 0.5;
            ldouble l1 = tlm_d1.l;
            ldouble j1 = l1 + s1;
            ldouble coeff = j1*(j1+1) - l1*(l1+1) - s1*(s1+1);
            dH(k1, k2) += 0.25*std::pow(137.035999139, -2)*dvdr*coeff*_o[ko]->getNorm(ir, *_g)*_o[k2]->getNorm(ir, *_g)*std::pow(r, 1)*dr;
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
  //std::cout << _c << std::endl;
}

ldouble SpinOrbitCorrection::getE0() {
  // not yet with correction!
  // TODO
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
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();
    int l2 = _o[k2]->l();
    int m2 = _o[k2]->m();
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

ldouble SpinOrbitCorrection::getE0Uncorrected() {
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
    int l1 = _o[k1]->l();
    int m1 = _o[k1]->m();
    int l2 = _o[k2]->l();
    int m2 = _o[k2]->m();
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

