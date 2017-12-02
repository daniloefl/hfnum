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
  //MatrixXld M = Ro[icl[0]] - Ri[icl[0]+1].inverse();
  VectorXld fm(M);
  for (int k = 0; k < M; ++k) fm(k) = 1.0;

  int idx = 0;

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
  for (int k = 0; k < _o.size(); ++k) {
    R[0] = Fm[1]*Fm[0].inverse();
  }

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
