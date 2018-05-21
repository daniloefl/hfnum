#include "LinearSystemBuilder.h"

#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include <iostream>

LinearSystemBuilder::LinearSystemBuilder(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om)
  : _g(g), _o(o), icl(i), _om(om) {
}

LinearSystemBuilder::~LinearSystemBuilder() {
}

void LinearSystemBuilder::prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<ldouble> &pot, std::vector<ldouble> &vsum_up, std::vector<ldouble> &vsum_dw) {
  int M = _om.N();
  int S = 0;
  for (int k = 0; k < _o.size(); ++k) {
    S += _om.N()*_g.N();
  }
  int Norb = _o.size();

  A.resize(S+Norb, S+Norb);
  b0.resize(S+Norb);
  A.setZero();
  b0.setZero();

  std::vector<Tr> TLA;

  int idxMatrix1 = 0;
  int idxMatrix2 = 0;
  ldouble a = 0;
  for (int idx1 = 0; idx1 < M; ++idx1) {
    int k1 = _om.orbital(idx1);
    int l1 = _om.l(idx1);
    ldouble l1_eq = _om.l(idx1);
    int m1 = _om.m(idx1);
    for (int i = 0; i < _g.N(); ++i) {
      idxMatrix1 = _g.N()*idx1 + i;
      if (_g.isLog()) {
        if (_om.s(idx1) > 0) {
          a = 2*std::pow(_g(i), 2)*(_o[k1]->E() - pot[i] - vsum_up[i]) - std::pow(l1_eq + 0.5, 2);
        } else {
          a = 2*std::pow(_g(i), 2)*(_o[k1]->E() - pot[i] - vsum_dw[i]) - std::pow(l1_eq + 0.5, 2);
        }
        TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))));
        TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(_g(i), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i)));
        b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))*(*_o[k1])(i);
      } else {
        if (_om.s(idx1) > 0) {
          a = 2*(_o[k1]->E() - pot[i] - vsum_up[i]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
        } else {
          a = 2*(_o[k1]->E() - pot[i] - vsum_dw[i]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
        }
        TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))));
        TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i)));
        b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))*(*_o[k1])(i);
      }

      if (i > 0) {
        if (_g.isLog()) {
          ldouble a = 0;
          if (_om.s(idx1) > 0) {
            a = 2*std::pow(_g(i-1), 2)*(_o[k1]->E() - pot[i-1] - vsum_up[i-1]) - std::pow(l1_eq + 0.5, 2);
          } else {
            a = 2*std::pow(_g(i-1), 2)*(_o[k1]->E() - pot[i-1] - vsum_dw[i-1]) - std::pow(l1_eq + 0.5, 2);
          }
          TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g(i-1), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i-1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i-1);
        } else {
          ldouble a = 0;
          if (_om.s(idx1) > 0) {
            a = 2*(_o[k1]->E() - pot[i-1] - vsum_up[i-1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          } else {
            a = 2*(_o[k1]->E() - pot[i-1] - vsum_dw[i-1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          }
          TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i-1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i-1);
        }

      }
      if (i < _g.N()-1) {
        if (_g.isLog()) {
          ldouble a = 0;
          if (_om.s(idx1) > 0) {
            a = 2*std::pow(_g(i+1), 2)*(_o[k1]->E() - pot[i+1] - vsum_up[i+1]) - std::pow(l1_eq + 0.5, 2);
          } else {
            a = 2*std::pow(_g(i+1), 2)*(_o[k1]->E() - pot[i+1] - vsum_dw[i+1]) - std::pow(l1_eq + 0.5, 2);
          }
          TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g(i+1), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i+1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i+1);
        } else {
          ldouble a = 0;
          if (_om.s(idx1) > 0) {
            a = 2*(_o[k1]->E() - pot[i+1] - vsum_up[i+1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          } else {
            a = 2*(_o[k1]->E() - pot[i+1] - vsum_dw[i+1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          }
          TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i+1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i+1);
        }

      }

      // sum psi^2 r^2 dr = 1
      ldouble dr = 0;
      if (i < _g.N() - 1) dr = _g(i+1) - _g(i);
      if (_g.isLog()) {
        b0(S+k1) += std::pow( (*_o[k1])(i)*std::pow(_g(i), -0.5)*_g(i), 2)*dr;
        TLA.push_back(Tr(S+k1, idxMatrix1, 2*(*_o[k1])(i)*_g(i)*dr));
      } else {
        b0(S+k1) += std::pow( (*_o[k1])(i)*_g(i), 2)*dr;
        TLA.push_back(Tr(S+k1, idxMatrix1, 2*(*_o[k1])(i)*std::pow(_g(i), 2)*dr));
      }
    }

    //TLA.push_back(Tr(S+Norb, S+k1, -2.0*o[k1].E()));
    b0(S+k1) += -1;
  }
  //TLA.push_back(Tr(S+Norb, S+Norb, 1.0));

  A.setFromTriplets(TLA.begin(), TLA.end());
}

void LinearSystemBuilder::prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<ldouble> &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap) {
  int M = _om.N();
  int S = 0;
  for (int k = 0; k < _o.size(); ++k) {
    S += _om.N()*_g.N();
  }
  int Norb = _o.size();
  int Nlam = lambda.size();

  // A x = b0
  // eqs. are:
  // for orbital k, index i: (12 - 10*(1 + (2*r^2*(E - pot[i] - Vd[i] + Vex[k1,k1][i]) - (l1+0.5)^2)*dx^2)/12) * o[k1][i]
  //                         + \sum_k2 10*(-2*r^2*Vex[k1,k2])*o[k2][i] + \sum_k2 10*(-2*r^2*lambda[k1,k2])*o[k2]
  //                         - (1 + 2*r(i-1)^2 * dx^2 * (E - pot[i-1] - Vd[i-1] + Vex[k1,k1][i-1]) - (l1+0.5)^2)/12 * o[k1][i-1]
  //                         - (1 + 2*r(i+1)^2 * dx^2 * (E - pot[i+1] - Vd[i+1] + Vex[k1,k1][i+1]) - (l1+0.5)^2)/12 * o[k1][i+1]
  //                         + \sum_k2 10*(-2*r^2*Vex[k1,k2])*o[k2][i-1] + \sum_k2 10*(-2*r^2*lambda[k1,k2])*o[k2]
  A.resize(S+Norb+Nlam, S+Norb+Nlam);
  b0.resize(S+Norb+Nlam);
  A.setZero();
  b0.setZero();

  std::vector<Tr> TLA;

  int idxMatrix1 = 0;
  int idxMatrix2 = 0;
  ldouble a = 0;
  for (int idx1 = 0; idx1 < M; ++idx1) {
    int k1 = _om.orbital(idx1);
    int l1 = _om.l(idx1);
    ldouble l1_eq = _om.l(idx1);
    int m1 = _om.m(idx1);
    for (int i = 0; i < _g.N(); ++i) {
      idxMatrix1 = _g.N()*idx1 + i;
      if (_g.isLog()) {
        a = 2*std::pow(_g(i), 2)*(_o[k1]->E() - pot[i] - vd[k1][i] + vex[std::pair<int,int>(k1,k1)][i]) - std::pow(l1_eq + 0.5, 2);
        TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))));
        TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(_g(i), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i)));
        b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))*(*_o[k1])(i);
      } else {
        a = 2*(_o[k1]->E() - pot[i] - vd[k1][i] + vex[std::pair<int,int>(k1,k1)][i]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
        TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))));
        TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i)));
        b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))*(*_o[k1])(i);
      }

      for (int idx2 = 0; idx2 < M; ++idx2) {
        if (idx2 == idx1) continue;
        int k2 = _om.orbital(idx2);
        int l2 = _om.l(idx2);
        int m2 = _om.m(idx2);
        idxMatrix2 = _g.N()*idx2 + i;
        if (_g.isLog()) {
          a = -2*std::pow(_g(i), 2)*vex[std::pair<int,int>(k1,k2)][i];
          TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(_g.dx(), 2)/12.0));
          b0(idxMatrix1) += 10.0*(a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i));
          if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
            int lidx = lambdaMap[100*idx1 + idx2];
            a = -2*std::pow(_g(i), 2);
            TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(_g.dx(), 2)/12.0*lambda[lidx]));
            TLA.push_back(Tr(idxMatrix1, S+Norb+lidx, 10.0*a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i)));
            b0(idxMatrix1) += 10.0*(a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i))*lambda[lidx];
          }
        } else {
          a = -2*vex[std::pair<int,int>(k1,k2)][i];
          TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(_g.dx(), 2)/12.0));
          b0(idxMatrix1) += 10.0*(a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i));
          if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
            int lidx = lambdaMap[100*idx1 + idx2];
            a = -2;
            TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(_g.dx(), 2)/12.0*lambda[lidx]));
            TLA.push_back(Tr(idxMatrix1, S+Norb+lidx, 10.0*a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i)));
            b0(idxMatrix1) += 10.0*(a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i)*lambda[lidx]);
          }
        }
      }

      if (i > 0) {
        if (_g.isLog()) {
          ldouble a = 2*std::pow(_g(i-1), 2)*(_o[k1]->E() - pot[i-1] - vd[k1][i-1] + vex[std::pair<int,int>(k1,k1)][i-1]) - std::pow(l1_eq + 0.5, 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g(i-1), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i-1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i-1);
        } else {
          ldouble a = 2*(_o[k1]->E() - pot[i-1] - vd[k1][i-1] + vex[std::pair<int,int>(k1,k1)][i-1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i-1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i-1);
        }

        for (int idx2 = 0; idx2 < M; ++idx2) {
          if (idx2 == idx1) continue;
          int k2 = _om.orbital(idx2);
          int l2 = _om.l(idx2);
          int m2 = _om.m(idx2);
          idxMatrix2 = _g.N()*idx2 + i;
          if (_g.isLog()) {
            a = -2*std::pow(_g(i-1), 2)*vex[std::pair<int,int>(k1,k2)][i-1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1);
            if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
              int lidx = lambdaMap[100*idx1 + idx2];
              a = -2*std::pow(_g(i-1), 2);
              TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(_g.dx(), 2)/12.0*lambda[lidx]));
              TLA.push_back(Tr(idxMatrix1, S+Norb+lidx, a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1)));
              b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1)*lambda[lidx];
            }
          } else {
            a = -2*vex[std::pair<int,int>(k1,k2)][i-1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1);
            if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
              int lidx = lambdaMap[100*idx1 + idx2];
              a = -2;
              TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(_g.dx(), 2)/12.0*lambda[lidx]));
              TLA.push_back(Tr(idxMatrix1, S+Norb+lidx, a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1)));
              b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1)*lambda[lidx];
            }
          }
        }

      }
      if (i < _g.N()-1) {
        if (_g.isLog()) {
          ldouble a = 2*std::pow(_g(i+1), 2)*(_o[k1]->E() - pot[i+1] - vd[k1][i+1] + vex[std::pair<int,int>(k1,k1)][i+1]) - std::pow(l1_eq + 0.5, 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g(i+1), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i+1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i+1);
        } else {
          ldouble a = 2*(_o[k1]->E() - pot[i+1] - vd[k1][i+1] + vex[std::pair<int,int>(k1,k1)][i+1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i+1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i+1);
        }

        for (int idx2 = 0; idx2 < M; ++idx2) {
          if (idx2 == idx1) continue;
          int k2 = _om.orbital(idx2);
          int l2 = _om.l(idx2);
          int m2 = _om.m(idx2);
          idxMatrix2 = _g.N()*idx2 + i;
          if (_g.isLog()) {
            a = -2*std::pow(_g(i+1), 2)*vex[std::pair<int,int>(k1,k2)][i+1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1);
            if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
              int lidx = lambdaMap[100*idx1 + idx2];
              a = -2*std::pow(_g(i+1), 2);
              TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(_g.dx(), 2)/12.0*lambda[lidx]));
              TLA.push_back(Tr(idxMatrix1, S+Norb+lidx, a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1)));
              b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1)*lambda[lidx];
            }
          } else {
            a = -2*vex[std::pair<int,int>(k1,k2)][i+1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1);
            if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
              int lidx = lambdaMap[100*idx1 + idx2];
              a = -2;
              TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(_g.dx(), 2)/12.0*lambda[lidx]));
              TLA.push_back(Tr(idxMatrix1, S+Norb+lidx, a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1)));
              b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1)*lambda[lidx];
            }
          }
        }

      }

      // sum psi^2 r^2 dr = 1
      ldouble dr = 0;
      if (i < _g.N() - 1) dr = _g(i+1) - _g(i);
      if (_g.isLog()) {
        b0(S+k1) += std::pow( (*_o[k1])(i)*std::pow(_g(i), -0.5)*_g(i), 2)*dr;
        TLA.push_back(Tr(S+k1, idxMatrix1, 2*(*_o[k1])(i)*_g(i)*dr));
      } else {
        b0(S+k1) += std::pow( (*_o[k1])(i)*_g(i), 2)*dr;
        TLA.push_back(Tr(S+k1, idxMatrix1, 2*(*_o[k1])(i)*std::pow(_g(i), 2)*dr));
      }

      // sum psi1 * psi2 r^2 dr = 0
      for (int idx2 = 0; idx2 < M; ++idx2) {
        int k2 = _om.orbital(idx2);
        int l2 = _om.l(idx2);
        if (k1 <= k2) continue;
        if (l1 != l2) continue;
        int lidx = lambdaMap[k1*100+k2];
        if (_g.isLog()) {
          TLA.push_back(Tr(S+Norb+lidx, _g.N()*idx1+i, (*_o[k2])(i)*std::pow((_g)(i), 2-1)*dr));
          TLA.push_back(Tr(S+Norb+lidx, _g.N()*idx2+i, (*_o[k1])(i)*std::pow((_g)(i), 2-1)*dr));
          b0(S+Norb+lidx) += (*_o[k1])(i)*(*_o[k2])(i)*std::pow((_g)(i), 2-1)*dr;
        } else {
          TLA.push_back(Tr(S+Norb+lidx, _g.N()*idx1+i, (*_o[k2])(i)*std::pow((_g)(i), 2)*dr));
          TLA.push_back(Tr(S+Norb+lidx, _g.N()*idx2+i, (*_o[k1])(i)*std::pow((_g)(i), 2)*dr));
          b0(S+Norb+lidx) += (*_o[k1])(i)*(*_o[k2])(i)*std::pow((_g)(i), 2)*dr;
        }
      }
    }

    //TLA.push_back(Tr(S+Norb, S+k1, -2.0*o[k1].E()));
    b0(S+k1) += -1;
  }
  //TLA.push_back(Tr(S+Norb, S+Norb, 1.0));

  A.setFromTriplets(TLA.begin(), TLA.end());
}

void LinearSystemBuilder::propagate(VectorXld &b, std::vector<ldouble> &dE, const ldouble gamma) {
  int M = _om.N();
  int S = 0;
  for (int k = 0; k < _o.size(); ++k) {
    S += _om.N()*_g.N();
  }
  for (int idx1 = 0; idx1 < M; ++idx1) {
    int k1 = _om.orbital(idx1);
    int l1 = _om.l(idx1);
    int m1 = _om.m(idx1);
    dE[k1] = -gamma*b(S+k1);
    for (int i = 0; i < _g.N(); ++i) {
      int idxMatrix1 = _g.N()*idx1 + i;
      (*_o[k1])(i) += -gamma*b(idxMatrix1);
    }
    _o[k1]->normalise(_g);
    idx1++;
  }
}

