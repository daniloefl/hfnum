#include "IterativeStandardMatrixSolver.h"

#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"
#include <iostream>

IterativeStandardMatrixSolver::IterativeStandardMatrixSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om)
  : _g(g), _o(o), icl(i), _om(om) {
}

IterativeStandardMatrixSolver::~IterativeStandardMatrixSolver() {
}


VectorXld IterativeStandardMatrixSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched) {
  int M = _om.N();

  std::vector<VectorXld> inward(_g.N());
  std::vector<VectorXld> outward(_g.N());
  matched.resize(_g.N());

  solveOutward(E, l, outward, Fm, Km);
  solveInward(E, l, inward, Fm, Km);

  match(matched, inward, outward);

  VectorXld F(M);
  F.setZero();
  //for (int k = 0; k < M; ++k) {
  //  int i = icl[k];
    int i = icl[M-1];
    F += (MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*matched[i] - (Fm[i+1]*matched[i+1]) - (Fm[i-1]*matched[i-1]);
  //}
  
  
  return F;
}



void IterativeStandardMatrixSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km) {
  int N = _g.N();
  int M = _om.N();
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
    solution[i].setZero();
  }
  for (int idx = 0; idx < M; ++idx) {
    ldouble Zeff = _o[idx]->n()*std::sqrt(std::fabs(2*E[idx]));
    solution[N-1](idx) = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(N-1)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(N-1)/((ldouble) _o[idx]->n()));
    solution[N-2](idx) = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(N-2)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(N-2)/((ldouble) _o[idx]->n()));
  }
  for (int i = N-2; i >= 1; --i) {
    JacobiSVD<MatrixXld> dec(Fm[i-1], ComputeThinU | ComputeThinV);
    solution[i-1] = dec.solve((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1]));
    //solution[i-1] = Km[i-1]*((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1]));
  }
}

void IterativeStandardMatrixSolver::solveOutward(std::vector<ldouble> &E, std::vector<int> &li, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km) {
  int N = _g.N();
  int M = _om.N();
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
    solution[i].setZero();
  }
  for (int idx = 0; idx < M; ++idx) {
    ldouble Zeff = _o[idx]->n()*std::sqrt(std::fabs(2*E[idx]));
    solution[0](idx) = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(0)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(0)/((ldouble) _o[idx]->n()));
    solution[1](idx) = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(1)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(1)/((ldouble) _o[idx]->n()));
  }
  for (int i = 1; i < N-1; ++i) {
    JacobiSVD<MatrixXld> dec(Fm[i+1], ComputeThinU | ComputeThinV);
    solution[i+1] = dec.solve((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
    //solution[i+1] = Km[i+1]*((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
  }
}
void IterativeStandardMatrixSolver::match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward) {
  int M = _om.N();
  for (int i = 0; i < _g.N(); ++i) {
    o[i].resize(M);
  }

  for (int idx = 0; idx < M; ++idx) {
    int k = _om.orbital(idx);
    int l = _om.l(idx);
    int m = _om.m(idx);
    ldouble ratio = outward[icl[M-1]](idx)/inward[icl[M-1]](idx);
    for (int i = 0; i < _g.N(); ++i) {
      if (i < icl[M-1]) {
        o[i](idx) = outward[i](idx);
      } else {
        o[i](idx) = ratio*inward[i](idx);
      }
    }
  }
}
