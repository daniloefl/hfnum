#include "LinearSystemBuilder.h"

#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"

LinearSystemBuilder::LinearSystemBuilder() {
}

LinearSystemBuilder::~LinearSystemBuilder() {
}


void LinearSystemBuilder::prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<Orbital> &o, std::vector<ldouble> &pot, std::map<int, Vd> &vd, std::map<std::pair<int, int>, Vex> &vex, const Grid &g) {
  int S = 0;
  for (int k = 0; k < o.size(); ++k) {
    S += (2*o[k].initialL()+1)*g.N();
  }
  int Norb = o.size();

  A.resize(S+Norb, S+Norb);
  b0.resize(S+Norb);
  A.setZero();
  b0.setZero();

  std::vector<Tr> TLA;

  int idx1 = 0;
  int idx2 = 0;
  int idxMatrix1 = 0;
  int idxMatrix2 = 0;
  ldouble a = 0;
  for (int k1 = 0; k1 < o.size(); ++k1) {
    for (int l1 = 0; l1 < o[k1].initialL()+1; ++l1) {
      for (int m1 = -l1; m1 < l1+1; ++m1) {

        for (int i = 0; i < g.N(); ++i) {
          idxMatrix1 = g.N()*idx1 + i;
          a = 2*std::pow(g(i), 2)*(o[k1].E() - pot[i] - vd[k1][std::pair<int,int>(l1, m1)][i]) - std::pow(l1 + 0.5, 2);
          TLA.push_back(Tr(idxMatrix1, idxMatrix1, (12.0 - 10.0*(1 + a*std::pow(g.dx(), 2)/12.0))));
          TLA.push_back(Tr(idxMatrix1, S+k1, -10.0*2*std::pow(g(i), 2)*std::pow(g.dx(), 2)/12.0*o[k1](i, l1, m1)));
          b0(idxMatrix1) += (12.0 - 10.0*(1 + a*std::pow(g.dx(), 2)/12.0))*o[k1](i, l1, m1);

          idx2 = 0;
          for (int k2 = 0; k2 < o.size(); ++k2) {
            for (int l2 = 0; l2 < o[k2].initialL()+1; ++l2) {
              for (int m2 = -l2; m2 < l2+1; ++m2) {
                idxMatrix2 = g.N()*idx2 + i;
                a = -2*std::pow(g(i), 2)*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i];
                TLA.push_back(Tr(idxMatrix1, idxMatrix2, 10.0*a*std::pow(g.dx(), 2)/12.0));
                b0(idxMatrix1) += 10.0*(a*std::pow(g.dx(), 2)/12.0*o[k2](i, l2, m2));
                idx2++;
              }
            }
          }

          if (i > 0) {
            ldouble a = 2*std::pow(g(i-1), 2)*(o[k1].E() - pot[i-1] - vd[k1][std::pair<int,int>(l1, m1)][i-1]) - std::pow(l1 + 0.5, 2);
            TLA.push_back(Tr(idxMatrix1, idxMatrix1-1, -(1 + a*std::pow(g.dx(), 2)/12.0)));
            TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(g(i-1), 2)*std::pow(g.dx(), 2)/12.0*o[k1](i-1, l1, m1)));
            b0(idxMatrix1) += -(1 + a*std::pow(g.dx(), 2)/12.0)*o[k1](i-1, l1, m1);

            idx2 = 0;
            for (int k2 = 0; k2 < o.size(); ++k2) {
              for (int l2 = 0; l2 < o[k2].initialL()+1; ++l2) {
                for (int m2 = -l2; m2 < l2+1; ++m2) {
                  idxMatrix2 = g.N()*idx2 + i;
                  a = -2*std::pow(g(i-1), 2)*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i-1];
                  TLA.push_back(Tr(idxMatrix1, idxMatrix2-1, a*std::pow(g.dx(), 2)/12.0));
                  b0(idxMatrix1) += a*std::pow(g.dx(), 2)/12.0*o[k2](i-1, l2, m2);
                  idx2++;
                }
              }
            }

          }
          if (i < g.N()-1) {
            ldouble a = 2*std::pow(g(i+1), 2)*(o[k1].E() - pot[i+1] - vd[k1][std::pair<int,int>(l1, m1)][i+1]) - std::pow(l1 + 0.5, 2);
            TLA.push_back(Tr(idxMatrix1, idxMatrix1+1, -(1 + a*std::pow(g.dx(), 2)/12.0)));
            TLA.push_back(Tr(idxMatrix1, S+k1, -2*std::pow(g(i+1), 2)*std::pow(g.dx(), 2)/12.0*o[k1](i+1, l1, m1)));
            b0(idxMatrix1) += -(1 + a*std::pow(g.dx(), 2)/12.0)*o[k1](i+1, l1, m1);

            idx2 = 0;
            for (int k2 = 0; k2 < o.size(); ++k2) {
              for (int l2 = 0; l2 < o[k2].initialL()+1; ++l2) {
                for (int m2 = -l2; m2 < l2+1; ++m2) {
                  idxMatrix2 = g.N()*idx2 + i;
                  a = -2*std::pow(g(i+1), 2)*vex[std::pair<int,int>(k1,k2)][std::pair<int,int>(l1, m1)][i+1];
                  TLA.push_back(Tr(idxMatrix1, idxMatrix2+1, a*std::pow(g.dx(), 2)/12.0));
                  b0(idxMatrix1) += a*std::pow(g.dx(), 2)/12.0*o[k2](i+1, l2, m2);
                  idx2++;
                }
              }
            }

          }

          // sum psi^2 r^2 dr = 1
          ldouble dr = 0;
          if (i < g.N() - 1) dr = g(i+1) - g(i);
          b0(S+k1) += std::pow( o[k1](i, l1, m1)*std::pow(g(i), -0.5)*g(i), 2)*dr;

          TLA.push_back(Tr(S+k1, idxMatrix1, 2*o[k1](i, l1, m1)*g(i)*dr));
        }

        idx1++;
      }
    }
    //TLA.push_back(Tr(S+Norb, S+k1, -2.0*o[k1].E()));
    b0(S+k1) += -1;
  }
  //TLA.push_back(Tr(S+Norb, S+Norb, 1.0));

  A.setFromTriplets(TLA.begin(), TLA.end());
}

void LinearSystemBuilder::propagate(VectorXld &b, std::vector<Orbital> &o, std::vector<ldouble> &dE, const Grid &g, const ldouble gamma) {
  int S = 0;
  for (int k = 0; k < o.size(); ++k) {
    S += (2*o[k].initialL()+1)*g.N();
  }
  int idx1 = 0;
  int idxMatrix1 = 0;
  for (int k1 = 0; k1 < o.size(); ++k1) {
    dE[k1] = -gamma*b(S+k1);

    for (int l1 = 0; l1 < o[k1].initialL()+1; ++l1) {
      for (int m1 = -l1; m1 < l1+1; ++m1) {

        for (int i = 0; i < g.N(); ++i) {
          idxMatrix1 = g.N()*idx1 + i;
          o[k1](i, l1, m1) += -gamma*b(idxMatrix1);
        }
        o[k1].normalise(g);
        idx1++;
      }
    }
  }
}

