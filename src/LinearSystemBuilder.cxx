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


void LinearSystemBuilder::prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<ldouble> &pot, std::map<int, Vd> &vd, std::map<std::pair<int, int>, Vex> &vex) {
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
        a = 2*std::pow(_g(i), 2)*(_o[k1]->E() - pot[i] - vd[k1][std::pair<int,int>(l1, m1)][i]) - std::pow(l1_eq + 0.5, 2);
        TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))));
        TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(_g(i), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i, l1, m1)));
        b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))*(*_o[k1])(i, l1, m1);
      } else {
        a = 2*(_o[k1]->E() - pot[i] - vd[k1][std::pair<int,int>(l1, m1)][i]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
        TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))));
        TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i, l1, m1)));
        b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(_g.dx(), 2)/12.0))*(*_o[k1])(i, l1, m1);
      }

      for (int idx2 = 0; idx2 < M; ++idx2) {
        int k2 = _om.orbital(idx2);
        int l2 = _om.l(idx2);
        int m2 = _om.m(idx2);
        idxMatrix2 = _g.N()*idx2 + i;
        if (_g.isLog()) {
          a = -2*std::pow(_g(i), 2)*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i];
          TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(_g.dx(), 2)/12.0));
          b0(idxMatrix1) += 10.0*(a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i, l2, m2));
        } else {
          a = -2*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i];
          TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(_g.dx(), 2)/12.0));
          b0(idxMatrix1) += 10.0*(a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i, l2, m2));
        }
      }

      if (i > 0) {
        if (_g.isLog()) {
          ldouble a = 2*std::pow(_g(i-1), 2)*(_o[k1]->E() - pot[i-1] - vd[k1][std::pair<int,int>(l1, m1)][i-1]) - std::pow(l1_eq + 0.5, 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g(i-1), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i-1, l1, m1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i-1, l1, m1);
        } else {
          ldouble a = 2*(_o[k1]->E() - pot[i-1] - vd[k1][std::pair<int,int>(l1, m1)][i-1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i-1, l1, m1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i-1, l1, m1);
        }

        for (int idx2 = 0; idx2 < M; ++idx2) {
          int k2 = _om.orbital(idx2);
          int l2 = _om.l(idx2);
          int m2 = _om.m(idx2);
          idxMatrix2 = _g.N()*idx2 + i;
          if (_g.isLog()) {
            a = -2*std::pow(_g(i-1), 2)*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i-1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1, l2, m2);
          } else {
            a = -2*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i-1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i-1, l2, m2);
          }
        }

      }
      if (i < _g.N()-1) {
        if (_g.isLog()) {
          ldouble a = 2*std::pow(_g(i+1), 2)*(_o[k1]->E() - pot[i+1] - vd[k1][std::pair<int,int>(l1, m1)][i+1]) - std::pow(l1_eq + 0.5, 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g(i+1), 2)*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i+1, l1, m1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i+1, l1, m1);
        } else {
          ldouble a = 2*(_o[k1]->E() - pot[i+1] - vd[k1][std::pair<int,int>(l1, m1)][i+1]) - (l1_eq+1)*l1_eq/std::pow(_g(i), 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(_g.dx(), 2)/12.0)));
          TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(_g.dx(), 2)/12.0*(*_o[k1])(i+1, l1, m1)));
          b0(idxMatrix1) += -(1 + a*std::pow(_g.dx(), 2)/12.0)*(*_o[k1])(i+1, l1, m1);
        }

        for (int idx2 = 0; idx2 < M; ++idx2) {
          int k2 = _om.orbital(idx2);
          int l2 = _om.l(idx2);
          int m2 = _om.m(idx2);
          idxMatrix2 = _g.N()*idx2 + i;
          if (_g.isLog()) {
            a = -2*std::pow(_g(i+1), 2)*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i+1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1, l2, m2);
          } else {
            a = -2*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i+1];
            TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(_g.dx(), 2)/12.0));
            b0(idxMatrix1) += a*std::pow(_g.dx(), 2)/12.0*(*_o[k2])(i+1, l2, m2);
          }
        }

      }

      // sum psi^2 r^2 dr = 1
      ldouble dr = 0;
      if (i < _g.N() - 1) dr = _g(i+1) - _g(i);
      if (_g.isLog()) {
        b0(S+k1) += std::pow( (*_o[k1])(i, l1, m1)*std::pow(_g(i), -0.5)*_g(i), 2)*dr;
        TLA.push_back(Tr(S+k1, idxMatrix1, 2*(*_o[k1])(i, l1, m1)*_g(i)*dr));
      } else {
        b0(S+k1) += std::pow( (*_o[k1])(i, l1, m1)*_g(i), 2)*dr;
        TLA.push_back(Tr(S+k1, idxMatrix1, 2*(*_o[k1])(i, l1, m1)*std::pow(_g(i), 2)*dr));
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
      (*_o[k1])(i, l1, m1) += -gamma*b(idxMatrix1);
    }
    _o[k1]->normalise(_g);
    idx1++;
  }
}

