#include "IterativeGreenSolver.h"

#include <iostream>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

IterativeGreenSolver::IterativeGreenSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om, ldouble Z)
  : _g(g), _o(o), icl(i), _om(om), _Z(Z) {

}

IterativeGreenSolver::~IterativeGreenSolver() {
}

void IterativeGreenSolver::setZ(ldouble Z) {
  _Z = Z;
}


VectorXld IterativeGreenSolver::solve(std::vector<ldouble> &E, Vradial &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched) {
  int M = _om.N();

  for (int idx = 0; idx < M; ++idx) {
    if (matched.find(idx) == matched.end()) matched.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (inward.find(idx) == inward.end()) inward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (outward.find(idx) == outward.end()) outward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (f.find(idx) == f.end()) f.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (homogeneousSolution.find(idx) == homogeneousSolution.end())homogeneousSolution.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (S.find(idx) == S.end()) S.insert(std::pair<int, Vradial>(idx, Vradial()));
  }
  for (int idx = 0; idx < M; ++idx) {
    matched[idx].resize(_g.N(), 0);
    inward[idx].resize(_g.N(), 0);
    outward[idx].resize(_g.N(), 0);
    f[idx].resize(_g.N(), 0);
    homogeneousSolution[idx].resize(_g.N(), 0);
    S[idx].resize(_g.N(), 0);
  }

  for (int idx = 0; idx < M; ++idx) {
    if (_g.isLog()) {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*(2*std::pow(_g(k), 2)*(E[idx] - pot[k] - vd[idx][k] + vex[std::pair<int,int>(idx,idx)][k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
      }
    } else {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*((E[idx] - pot[k] - vd[idx][k] + vex[std::pair<int,int>(idx,idx)][k]) - std::pow(_o[idx]->l() + 0.5, 2));
      }
    }
  }

  VectorXld W(M);
  W.setZero();

  // solve in direct order
  for (int idx = 0; idx < M; ++idx) {
    solveOutward(E, idx, outward[idx]);
    solveInward(E, idx, inward[idx]);
    match(idx, homogeneousSolution[idx], inward[idx], outward[idx]);
    normalise(idx, homogeneousSolution[idx]);
    W(idx) = (inward[idx][icl[idx]+1] - inward[idx][icl[idx]-1])*outward[idx][icl[idx]]/(2*_g.dx());
    W(idx) -= (outward[idx][icl[idx]+1] - outward[idx][icl[idx]-1])*inward[idx][icl[idx]]/(2*_g.dx());
  }

  // calculate independent term S
  for (int idx1 = 0; idx1 < M; ++idx1) {
    for (int k = 0; k < _g.N(); ++k) {
      for (int idx2 = 0; idx2 < M; ++idx2) {
        if (idx1 == idx2) continue;
        if (_g.isLog()) {
          S[idx1][k] += 2*std::pow(_g(k), 2)*vex[std::pair<int,int>(idx1, idx2)][k]*homogeneousSolution[idx2][k];
          if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
            int lidx = lambdaMap[100*idx1 + idx2];
            S[idx1][k] += 2*std::pow(_g(k), 2)*lambda[lidx]*homogeneousSolution[idx2][k];
          }
        } else {
          S[idx1][k] += vex[std::pair<int,int>(idx1, idx2)][k]*homogeneousSolution[idx2][k];
          if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
            int lidx = lambdaMap[100*idx1 + idx2];
            S[idx1][k] += lambda[lidx]*homogeneousSolution[idx2][k];
          }
        }
      }
    }
  }
  // do the following for the inward and outward solutions and sum it
  for (int idx1 = 0; idx1 < M; ++idx1) {
    // sum(k) int_0^k homogeneousSolution[idx1][j]*S(j) dj
    // the solution to the inhomogeneous equation is
    // matched[idx1][k] = sum(k)*homogeneousSolution[idx1][k]
    ldouble sum = 0.0L;
    for (int k = 0; k < _g.N(); ++k) {
      if (_g.isLog()) {
        sum += outward[idx1][k]*S[idx1][k]*_g.dx()*_g(k);
      } else {
        sum += outward[idx1][k]*S[idx1][k]*_g.dx();
      }
      matched[idx1][k] += sum*inward[idx1][k];
    }
  }
  // now for the outward, but weigh it by the Wronskian to match it at icl
  for (int idx1 = 0; idx1 < M; ++idx1) {
    ldouble sum = 0.0L;
    for (int k = _g.N()-1; k >= 0; --k) {
      if (_g.isLog()) {
        sum += inward[idx1][k]*S[idx1][k]*_g.dx()*_g(k);
      } else {
        sum += inward[idx1][k]*S[idx1][k]*_g.dx();
      }
      matched[idx1][k] += sum*outward[idx1][k];
    }
  }

  // add homogeneous solution
  VectorXld Ssum(M);
  Ssum.setZero();
  for (int idx1 = 0; idx1 < M; ++idx1) {
    for (int k = 0; k < _g.N(); ++k) {
      Ssum(idx1) += S[idx1][k];
    }
  }
  for (int idx1 = 0; idx1 < M; ++idx1) {
    if (Ssum(idx1) == 0) {
      for (int k = 0; k < _g.N(); ++k) {
        matched[idx1][k] = homogeneousSolution[idx1][k];
      }
    }
  }

  for (int idx = 0; idx < M; ++idx) {
    normalise(idx, matched[idx]);
  }

  return W;
}

VectorXld IterativeGreenSolver::solve(std::vector<ldouble> &E, Vradial &pot, Vradial &vup, Vradial &vdw, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched) {
  int M = _om.N();

  for (int idx = 0; idx < M; ++idx) {
    if (matched.find(idx) == matched.end()) matched.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (inward.find(idx) == inward.end()) inward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (outward.find(idx) == outward.end()) outward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (f.find(idx) == f.end()) f.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (homogeneousSolution.find(idx) == homogeneousSolution.end())homogeneousSolution.insert(std::pair<int, Vradial>(idx, Vradial()));
  }
  for (int idx = 0; idx < M; ++idx) {
    matched[idx].resize(_g.N(), 0);
    inward[idx].resize(_g.N(), 0);
    outward[idx].resize(_g.N(), 0);
    f[idx].resize(_g.N(), 0);
    homogeneousSolution[idx].resize(_g.N(), 0);
  }

  for (int idx = 0; idx < M; ++idx) {
    if (_g.isLog()) {
      for (int k = 0; k < _g.N(); ++k) {
        if (_o[idx]->term().find('+') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*(2*std::pow(_g(k), 2)*(E[idx] - pot[k] - vup[k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
        if (_o[idx]->term().find('-') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*(2*std::pow(_g(k), 2)*(E[idx] - pot[k] - vdw[k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
      }
    } else {
      for (int k = 0; k < _g.N(); ++k) {
        if (_o[idx]->term().find('+') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*((E[idx] - pot[k] - vup[k]) - std::pow(_o[idx]->l() + 0.5, 2));
        if (_o[idx]->term().find('-') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*((E[idx] - pot[k] - vdw[k]) - std::pow(_o[idx]->l() + 0.5, 2));
      }
    }
  }

  VectorXld W(M);
  W.setZero();
  for (int idx = 0; idx < M; ++idx) {
    solveOutward(E, idx, outward[idx]);
    solveInward(E, idx, inward[idx]);
    match(idx, matched[idx], inward[idx], outward[idx]);
    normalise(idx, matched[idx]);
    W(idx) = (inward[idx][icl[idx]+1] - inward[idx][icl[idx]-1])*outward[idx][icl[idx]]/(2*_g.dx());
    W(idx) -= (outward[idx][icl[idx]+1] - outward[idx][icl[idx]-1])*inward[idx][icl[idx]]/(2*_g.dx());
  }

  //VectorXld F(M);
  //// calculate first derivative in icl[idx]
  //for (int idx = 0; idx < M; ++idx) {
  //  F(idx) = ( (12 - 10*f[idx][icl[idx]])*matched[idx][icl[idx]] 
  //                  - f[idx][icl[idx]-1]*matched[idx][icl[idx]-1]
  //                  - f[idx][icl[idx]+1]*matched[idx][icl[idx]+1]
  //                );
  //}

  return W;
}



void IterativeGreenSolver::solveInward(std::vector<ldouble> &E, int idx, Vradial &solution) {
  int N = _g.N();
  solution.resize(N);
  ldouble a0 = std::sqrt(2*std::fabs(E[idx]))/_Z;
  solution[N-1] = std::pow(_g(N-1), 0.5)*std::exp(-_g(N-1)*a0);
  solution[N-2] = std::pow(_g(N-2), 0.5)*std::exp(-_g(N-2)*a0);
  for (int k = N-2; k >= 1; --k) {
    solution[k-1] = ((12 - f[idx][k]*10)*solution[k] - f[idx][k+1]*solution[k+1])/f[idx][k-1];
  }
}

void IterativeGreenSolver::solveOutward(std::vector<ldouble> &E, int idx, Vradial &solution) {
  int N = _g.N();
  solution.resize(N);
  ldouble a0 = std::sqrt(2*std::fabs(E[idx]))/_Z;
  solution[0] = std::pow(_g(0), _o[idx]->l() + 0.5);
  solution[1] = std::pow(_g(1), _o[idx]->l() + 0.5);
  if (_o[idx]->n() == 1) {
    solution[0] = std::pow(_g(0), 0.5)*std::exp(-_g(0)/a0);
    solution[1] = std::pow(_g(1), 0.5)*std::exp(-_g(1)/a0);
  } else if (_o[idx]->n() == 2 && _o[idx]->l() == 0) {
    solution[0] = std::pow(_g(0), 0.5)*(2 - _g(0)/a0)*std::exp(-_g(0)/(2.0*a0));
    solution[1] = std::pow(_g(1), 0.5)*(2 - _g(1)/a0)*std::exp(-_g(1)/(2.0*a0));
  } else if (_o[idx]->n() == 2 && _o[idx]->l() == 1) {
    solution[0] = std::pow(_g(0), 1.5)/a0*std::exp(-_g(0)/(2.0*a0));
    solution[1] = std::pow(_g(1), 1.5)/a0*std::exp(-_g(1)/(2.0*a0));
  }
  if (_i0.size() > idx && _i1.size() > idx) {
    solution[0] = std::pow(_g(0), 0.5)*_i0[idx];
    solution[1] = std::pow(_g(1), 0.5)*_i1[idx];
  }
  for (int k = 1; k < N-2; ++k) {
    solution[k+1] = ((12.0 - f[idx][k]*10.0)*solution[k] - f[idx][k-1]*solution[k-1])/f[idx][k+1];
  }
}

void IterativeGreenSolver::normalise(int k, Vradial &o) {
  ldouble norm = 0;
  for (int i = 0; i < _g.N(); ++i) {
    ldouble r = _g(i);
    ldouble dr = 0;
    if (i < _g.N()-1) dr = std::fabs(_g(i+1) - _g(i));
    ldouble ov = o[i];
    if (_g.isLog()) ov *= std::pow(r, -0.5);
    //o_untransformed[i] = ov;
    norm += std::pow(ov*r, 2)*dr;
  }
  for (int i = 0; i < _g.N(); ++i) {
    o[i] /= std::sqrt(norm);
  }
}

void IterativeGreenSolver::match(int k, Vradial &o, Vradial &inward, Vradial &outward) {
  o.resize(_g.N());
  ldouble ratio = outward[icl[k]]/inward[icl[k]];
  for (int i = 0; i < _g.N(); ++i) {
    if (i < icl[k]) {
      o[i] = outward[i];
    } else {
      o[i] = ratio*inward[i];
    }
  }

}
