#include "IterativeRenormalisedSolver.h"

#include <iostream>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"

IterativeRenormalisedSolver::IterativeRenormalisedSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om)
  : _g(g), _o(o), icl(i), _om(om) {
  kl = 0;
}

IterativeRenormalisedSolver::~IterativeRenormalisedSolver() {
}

VectorXld IterativeRenormalisedSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched, std::vector<int> &nodes) {
  int M = _om.N();
  kl = _o.size()-1;

  std::vector<MatrixXld> Ri(_g.N());
  std::vector<MatrixXld> Ro(_g.N());
  std::vector<VectorXld> fix_inward(_g.N());
  std::vector<VectorXld> fix_outward(_g.N());
  matched.resize(_g.N());

  solveOutward(E, l, Fm, Km, Ro);
  solveInward(E, l, Fm, Km, Ri);

  // originally the paper proposes to use the determinant
  // however, if there is perfect agreement in some orbitals
  // (for example, some orbitals are zero, due to useless added spherical harmonics components)
  // then the determinant is dragged to zero
  // this happens because the determinant is the product of the singular values and
  // some of the singular values are zero simply because the full solution is zero
  //ldouble F = std::fabs((Ro[icl[kl]] - Ri[icl[kl]+1].inverse()).determinant());
  //A - B^-1 = B^-1 (B A - I)
  //det(A - B^-1) = 0 => det(BA - I) = 0
  VectorXld F(_o.size());
  for (int idx = 0; idx < M; ++idx) {
    MatrixXld A = Ro[icl[idx]];
    MatrixXld B = Ri[icl[idx]+1];
    F(idx) = ((B*A - MatrixXld::Identity(M,M)).determinant());
    //F(idx) = ((Ro[icl[idx]] - Ri[icl[idx]+1].inverse()).determinant());
  }

  MatrixXld Mm = Ro[icl[kl]] - Ri[icl[kl]+1].inverse();
  VectorXld fm(M);
  fm.setZero();
  JacobiSVD<MatrixXld> dec_Mm(Mm, ComputeThinU | ComputeThinV);
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    if (dec_Mm.singularValues()(idx) != 0) {
      fm += dec_Mm.matrixV().block(0, idx, M, 1)/dec_Mm.singularValues()(idx);
    }
  }

  fix_outward[icl[kl]] = fm;
  for (int i = icl[kl]-1; i >= 0; --i) {
    fix_outward[i] = Ro[i].inverse()*fix_outward[i+1];
  }

  fix_inward[icl[kl]] = fm;
  for (int i = icl[kl]+1; i < _g.N(); ++i) {
    if (i == _g.N()-1) {
      fix_inward[i].resize(M, 1);
      fix_inward[i].setZero();
    } else {
      fix_inward[i] = Ri[i].inverse()*fix_inward[i-1];
    }
  }

  for (int i = icl[kl]; i >= 0; --i) {
    fix_outward[i] = Fm[i].inverse()*fix_outward[i];
  }
  for (int i = icl[kl]; i < _g.N(); ++i) {
    fix_inward[i] = Fm[i].inverse()*fix_inward[i];
  }
  match(matched, fix_inward, fix_outward);

  nodes.clear();
  nodes.push_back(0);
  for (int i = icl[kl]-1; i >= 3; --i) {
    if (Ro[i].determinant() < 0)
      nodes[0]++;
  }
  for (int i = icl[kl]+1; i < _g.N()-3; ++i) {
    if (Ri[i].determinant() < 0)
      nodes[0]++;
  }

  return F;
}

void IterativeRenormalisedSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R) {
  int N = _g.N();
  int M = _om.N();
  R.resize(N);
  for (int i = 0; i < N; ++i) {
    R[i].resize(M, M);
  }
  R[N-1].setZero();
  for (int i = N-1; i >= 1; --i) {
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
  //MatrixXld psi0(M,M);
  //MatrixXld psi1(M,M);
  //psi0.setZero();
  //psi1.setZero();
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    if (l == 0)
      R[0](idx, idx) = 6/_g.dx() - 5 + (1 - E[k])*_g.dx();
    if (l == 1)
      R[0](idx, idx) = - 5 + 1.5*_g.dx();
  }
  for (int i = 1; i <= N-1; ++i) {
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
    ldouble ratio = outward[icl[kl]](idx)/inward[icl[kl]](idx);
    for (int i = 0; i < _g.N(); ++i) {
      if (i < icl[kl]) {
        o[i](idx) = outward[i](idx);
      } else {
        o[i](idx) = ratio*inward[i](idx);
      }
    }
  }

}
