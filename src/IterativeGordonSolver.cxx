#include "IterativeGordonSolver.h"

#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

IterativeGordonSolver::IterativeGordonSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om)
  : _g(g), _o(o), icl(i), _om(om) {
}

IterativeGordonSolver::~IterativeGordonSolver() {
}


ldouble IterativeGordonSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched) {
  int M = _om.N();

  std::vector< std::vector<VectorXld> > inward(M);
  std::vector< std::vector<VectorXld> > outward(M);
  std::vector<VectorXld> fix_inward(_g.N());
  std::vector<VectorXld> fix_outward(_g.N());
  matched.resize(_g.N());

  for (int idx = 0; idx < M; ++idx) {
    inward[idx] = std::vector<VectorXld>(_g.N());
    outward[idx] = std::vector<VectorXld>(_g.N());
    solveOutward(E, l, outward[idx], Fm, Km, idx);
    solveInward(E, l, inward[idx], Fm, Km, idx);
  }
  MatrixXld D(2*M, 2*M);
  MatrixXld Da(M, M);
  MatrixXld Db(M, M);
  MatrixXld Dc(M, M);
  MatrixXld Dd(M, M);
  for (int idx1 = 0; idx1 < M; ++idx1) {
    int k = _om.orbital(idx1);
    int l = _om.l(idx1);
    int m = _om.m(idx1);
    int icl0 = icl[k];
    for (int idx2 = 0; idx2 < M; ++idx2) {
      D(idx1, idx2) = outward[idx2][icl0](idx1);
      Da(idx1, idx2) = outward[idx2][icl0](idx1);

      D(idx1, M + idx2) = inward[idx2][icl0](idx1);
      Db(idx1, idx2) = inward[idx2][icl0](idx1);

      D(M+idx1, idx2) = outward[idx2][icl0+1](idx1);
      Dc(idx1, idx2) = outward[idx2][icl0+1](idx1);

      D(M+idx1, M+idx2) = inward[idx2][icl0+1](idx1);
      Dd(idx1, idx2) = inward[idx2][icl0+1](idx1);
    }
  }
  ldouble F = D.determinant();
  VectorXld left(M);
  for (int i = 0; i < M; ++i) left(i) = 1.0;
  VectorXld r(M);
  r = (Db+Dd).inverse()*((Da+Dc)*left);

  for (int i = 0; i < _g.N(); ++i) {
    fix_inward[i].resize(M);
    fix_outward[i].resize(M);
    fix_inward[i].setZero();
    fix_outward[i].setZero();
  }

  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    int icl0 = icl[k];
    for (int i = 0; i < _g.N(); ++i) {
      if (i <= icl0+1) {
        for (int ks = 0; ks < M; ++ks) {
          fix_outward[i](idx) += outward[ks][i](idx)*left(ks);
        }
      }
      if (i >= icl0-1) {
        for (int ks = 0; ks < M; ++ks) {
          fix_inward[i](idx) += inward[ks][i](idx)*r(ks);
        }
      }
    }
  }
  match(matched, fix_inward, fix_outward);

  return std::fabs(F);
}



void IterativeGordonSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, int k_init) {
  int N = _g.N();
  int M = _om.N();
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
  }
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    solution[N-1](idx) = 0;
    solution[N-2](idx) = 1;
    if (idx == k_init) solution[N-2](idx) *= 2;
  }
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    for (int i = N-2; i >= icl[k]-1; --i) {
      //JacobiSVD<MatrixXld> dec(Fm[i-1], ComputeThinU | ComputeThinV);
      //solution[i-1] = dec.solve((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1]));
      solution[i-1] = Km[i-1]*((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1])); 
    }
  }
}

void IterativeGordonSolver::solveOutward(std::vector<ldouble> &E, std::vector<int> &li, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, int k_init) {
  int N = _g.N();
  int M = _om.N();
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
  }
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    solution[0](idx) = 0;
    solution[1](idx) = 1;
    if ((_o[k]->initialN() - _o[k]->initialL() - 1) % 2 == 1) {
      solution[0](idx) *= -1;
      solution[1](idx) *= -1;
    }
    if (idx == k_init) solution[1](idx) *= 2;
  }
  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    for (int i = 1; i <= icl[k]+1; ++i) {
      //JacobiSVD<MatrixXld> dec(Fm[i+1], ComputeThinU | ComputeThinV);
      //solution[i+1] = dec.solve((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
      solution[i+1] = Km[i+1]*((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
    }
  }
}
void IterativeGordonSolver::match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward) {
  int M = _om.N();
  for (int i = 0; i < _g.N(); ++i) {
    o[i].resize(M);
  }

  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    ldouble ratio = outward[icl[k]](idx)/inward[icl[k]](idx);
    for (int i = 0; i < _g.N(); ++i) {
      if (i < icl[k]) {
        o[i](idx) = outward[i](idx);
      } else {
        o[i](idx) = ratio*inward[i](idx);
      }
    }
  }
}
