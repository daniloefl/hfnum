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

ldouble IterativeRenormalisedSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched) {
  int M = _om.N();
  kl = icl.size()-1;

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
  MatrixXld A = Ro[icl[kl]];
  MatrixXld B = Ri[icl[kl]+1];
  ldouble F = (B*A - MatrixXld::Identity(M,M)).determinant();
  /*
  //JacobiSVD<MatrixXld> dec_BA(B*A - MatrixXld::Identity(M,M), ComputeThinU | ComputeThinV);
  JacobiSVD<MatrixXld> dec_BA(B*A, ComputeThinU | ComputeThinV);
  std::vector<ldouble> svd;
  for (int idx = 0; idx < M; ++idx) {
    ldouble s = dec_BA.singularValues()(idx);
    //if (s < 0) s = -s;
    svd.push_back(s);
  }
  ldouble F = 11;
  for (int idx = 0; idx < M; ++idx) {
    F *= (svd[idx] - 1);
  }*/

  MatrixXld Mm = Ro[icl[kl]] - Ri[icl[kl]+1].inverse();
  VectorXld fm(M);
  fm.setZero();
  JacobiSVD<MatrixXld> dec_Mm(Mm, ComputeThinU | ComputeThinV);
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    bool isPrimary = (l == _o[k]->initialL() && m == _o[k]->initialM());
    if (dec_Mm.singularValues()(idx) != 0) {
      fm += dec_Mm.matrixV().block(0, idx, M, 1)/dec_Mm.singularValues()(idx);
    }
  }
  //std::cout << "Mm:" << std::endl << Mm << std::endl;
  //std::cout << "Mm SV:" << std::endl << dec_Mm.singularValues() << std::endl;
  //std::cout << "Mm right-singular vectors:" << std::endl << dec_Mm.matrixV() << std::endl;
  //std::cout << "fm:" << std::endl << fm << std::endl;

  // this is the determinant, but scale it so that we avoid numerical errors
  //ldouble F = 0;
  //for (int idx = 0; idx < M; ++idx) {
  //  F += std::log(std::fabs(dec_Mm.singularValues()(idx)));
  //}
  //if (first) {
  //  first = false;
  //  shiftF = -F; // use first calculation to shift F to zero and avoid numerical errors in the next iteration
  //}
  //F += shiftF;
  //F = std::exp(F); // if this is commented out, the minimum is at - infinity, so F must be globally minimised and the minimum cannot be approximated with a paraboloid

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

  return F; //std::fabs(F);
}

void IterativeRenormalisedSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R) {
  int N = _g.N();
  int M = _om.N();
  R.resize(N);
  for (int i = 0; i < N; ++i) {
    R[i].resize(M, M);
  }
  R[N-1].setZero();
  for (int i = N-1; i >= icl[kl]-1; --i) {
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
      if (l == _o[k]->initialL() && m == _o[k]->initialM() && idx == idx2) psi1(idx, idx2) *= 10;
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
  for (int i = 1; i <= icl[kl]+1; ++i) {
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
