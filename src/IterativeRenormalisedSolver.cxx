#include "IterativeRenormalisedSolver.h"

#include <iostream>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"

IterativeRenormalisedSolver::IterativeRenormalisedSolver(const Grid &g, std::vector<Orbital> &o, std::vector<int> &i, OrbitalMapper &om)
  : _g(g), _o(o), icl(i), _om(om) {
}

IterativeRenormalisedSolver::~IterativeRenormalisedSolver() {
}


ldouble IterativeRenormalisedSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched) {
  int M = _om.N();

  std::vector<MatrixXld> Ri(_g.N());
  std::vector<MatrixXld> Ro(_g.N());
  std::vector<VectorXld> fix_inward(_g.N());
  std::vector<VectorXld> fix_outward(_g.N());
  matched.resize(_g.N());

  solveOutward(E, l, Fm, Km, Ro);
  solveInward(E, l, Fm, Km, Ri);

  ldouble F = (Ro[icl[0]] - Ri[icl[0]+1].inverse()).determinant();
  MatrixXld Mm = Ro[icl[0]] - Ri[icl[0]+1].inverse();
  VectorXld fm(M);
  fm.setZero();
  JacobiSVD<MatrixXld> dec_Mm(Mm, ComputeThinU | ComputeThinV);
  ldouble limit_SV = 1e-2;
  for (int idx = 0; idx < M; ++idx) {
    if (std::fabs(dec_Mm.singularValues()(idx)) < limit_SV) {
      fm += dec_Mm.matrixV().block(0, idx, M, 1)/dec_Mm.singularValues()(idx);
    }
  }
  //std::cout << "Mm:" << std::endl << Mm << std::endl;
  //std::cout << "Mm SV:" << std::endl << dec_Mm.singularValues() << std::endl;
  //std::cout << "Mm right-singular vectors:" << std::endl << dec_Mm.matrixV() << std::endl;
  //std::cout << "fm:" << std::endl << fm << std::endl;

  fix_outward[icl[0]] = fm;
  for (int i = icl[0]-1; i >= 0; --i) {
    fix_outward[i] = Ro[i].inverse()*fix_outward[i+1];
  }

  fix_inward[icl[0]] = fm;
  for (int i = icl[0]+1; i < _g.N(); ++i) {
    if (i == _g.N()-1) {
      fix_inward[i].resize(M, 1);
      fix_inward[i].setZero();
    } else {
      fix_inward[i] = Ri[i].inverse()*fix_inward[i-1];
    }
  }

  for (int i = icl[0]; i >= 0; --i) {
    fix_outward[i] = Fm[i].inverse()*fix_outward[i];
  }
  for (int i = icl[0]; i < _g.N(); ++i) {
    fix_inward[i] = Fm[i].inverse()*fix_inward[i];
  }
  match(matched, fix_inward, fix_outward);

  return std::fabs(F);
}

void IterativeRenormalisedSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R) {
  int N = _g.N();
  int M = _om.N();
  R.resize(N);
  for (int i = 0; i < N; ++i) {
    R[i].resize(M, M);
  }
  R[N-1].setZero();
  for (int i = N-1; i >= icl[0]-1; --i) {
    R[i-1] = Km[i-1]*(MatrixXld::Identity(M, M)*12 - Fm[i-1]*10);
    if (i < N-1 && R[i].determinant() != 0) R[i-1] -= R[i].inverse();
  }
}

void IterativeRenormalisedSolver::solveOutward(std::vector<ldouble> &E, std::vector<int> &li, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R) {
  int N = _g.N();
  int M = _om.N();
  R.resize(N);
  for (int i = 0; i < N; ++i) {
    R[i].resize(M, M);
  }
  R[0].setZero();
  MatrixXld psi0(M,M);
  MatrixXld psi1(M,M);
  psi0.setZero();
  psi1.setZero();
  for (int idx2 = 0; idx2 < M; ++idx2) {
    for (int idx = 0; idx < M; ++idx) {
      int k = _om.orbital(idx);
      int l = _om.l(idx);
      int m = _om.m(idx);
      psi0(idx, idx) = 1e-2;
      psi1(idx, idx) = 1e-1;
      if (l == _o[k].initialL() && m == _o[k].initialM() && idx == idx2) psi1(idx, idx2) *= 10;
    }
  }
  R[0] = Fm[1]*psi1*(Fm[0]*psi0).inverse();
  /*
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    if (l == 0)
      R[0](idx, idx) = 6/_g.dx() - 5 + (1 - E[k])*_g.dx();
    if (l == 1)
      R[0](idx, idx) = - 5 + 1.5*_g.dx();
  }*/
  for (int i = 1; i <= icl[0]+1; ++i) {
    R[i] = Km[i]*(MatrixXld::Identity(M, M)*12 - Fm[i]*10);
    if (R[i-1].determinant() != 0) R[i] -= R[i-1].inverse();
  }
}
void IterativeRenormalisedSolver::match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward) {
  int M = _om.N();
  for (int i = 0; i < _g.N(); ++i) {
    o[i].resize(M);
  }

  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    ldouble ratio = outward[icl[0]](idx)/inward[icl[0]](idx);
    for (int i = 0; i < _g.N(); ++i) {
      if (i < icl[0]) {
        o[i](idx) = outward[i](idx);
      } else {
        o[i](idx) = ratio*inward[i](idx);
      }
    }
  }

}
