#include "IterativeStandardSolver.h"

#include <iostream>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

IterativeStandardSolver::IterativeStandardSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om)
  : _g(g), _o(o), icl(i), _om(om) {

}

IterativeStandardSolver::~IterativeStandardSolver() {
}


ldouble IterativeStandardSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::map<int, Vradial> &matched) {
  int M = _om.N();
  ldouble F = 0;

  for (int idx = 0; idx < M; ++idx) {
    if (matched.find(idx) == matched.end()) matched.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (inward.find(idx) == inward.end()) inward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (outward.find(idx) == outward.end()) outward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (f.find(idx) == f.end()) f.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (s.find(idx) == s.end()) s.insert(std::pair<int, Vradial>(idx, Vradial()));
  }
  for (int idx = 0; idx < M; ++idx) {
    matched[idx].resize(_g.N(), 0);
    inward[idx].resize(_g.N(), 0);
    outward[idx].resize(_g.N(), 0);
    f[idx].resize(_g.N(), 0);
    s[idx].resize(_g.N(), 0);
  }

  for (int idx = 0; idx < M; ++idx) {
    if (_g.isLog()) {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*2*std::pow(_g(k), 2)*(E[idx] - vd[idx][k]) - std::pow(l[idx] - 0.5, 2);
      }
    } else {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*(E[idx] - vd[idx][k]) - std::pow(l[idx] - 0.5, 2);
      }
    }
  }

  // solve in direct order
  for (int idx = 0; idx < M; ++idx) {
    solveOutward(E, l, vd, vex, matched, idx, outward[idx]);
    solveInward(E, l, vd, vex, matched, idx, inward[idx]);
    match(idx, matched[idx], inward[idx], outward[idx]);

    for (int idx1 = 0; idx1 < M; ++idx1) {
      std::fill(s[idx1].begin(), s[idx1].end(), 0);
      for (int idx2 = 0; idx2 < M; ++idx2) {
        for (int k = 0; k < _g.N(); ++k) {
          if (_g.isLog()) {
            s[idx1][k] += std::pow(_g.dx(), 2)/12.0*2*std::pow(_g(k), 2)*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
          } else {
            s[idx1][k] += std::pow(_g.dx(), 2)/12.0*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
          }
        }
      }

    }
  }

  // solve in inverse order
  for (int idx = M-1; idx >= 0; --idx) {
    solveOutward(E, l, vd, vex, matched, idx, outward[idx]);
    solveInward(E, l, vd, vex, matched, idx, inward[idx]);
    match(idx, matched[idx], inward[idx], outward[idx]);

    for (int idx1 = 0; idx1 < M; ++idx1) {
      std::fill(s[idx1].begin(), s[idx1].end(), 0);
      for (int idx2 = 0; idx2 < M; ++idx2) {
        for (int k = 0; k < _g.N(); ++k) {
          if (_g.isLog()) {
            s[idx1][k] += std::pow(_g.dx(), 2)/12.0*2*std::pow(_g(k), 2)*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
          } else {
            s[idx1][k] += std::pow(_g.dx(), 2)/12.0*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
          }
        }
      }

    }
  }

  // calculate first derivative in icl[idx]
  for (int idx = 0; idx < M; ++idx) {
    F += std::fabs( (12 - 10*f[idx][icl[idx]])*matched[idx][icl[idx]] 
                    - f[idx][icl[idx]-1]*matched[idx][icl[idx]-1]
                    - f[idx][icl[idx]+1]*matched[idx][icl[idx]+1]
                    + s[idx][icl[idx]-1]
                    + s[idx][icl[idx]]
                    + s[idx][icl[idx]+1]
                  );
  }

  return F;
}



void IterativeStandardSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::map<int, Vradial> &matched, int idx, Vradial &solution) {
  int N = _g.N();
  solution.resize(N);
  solution[N-1] = std::exp(-std::sqrt(2*std::fabs(E[idx]))*_g(N-1));
  solution[N-2] = std::exp(-std::sqrt(2*std::fabs(E[idx]))*_g(N-2));
  for (int k = N-2; k >= 0; --k) {
    solution[k-1] = ((12 - f[idx][k]*10)*solution[k] - f[idx][k+1]*solution[k+1] + s[idx][k])/f[idx][k-1];
    //std::cout << "inward " << solution[k-1] << std::endl;
  }
}

void IterativeStandardSolver::solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::map<int, Vradial> &matched, int idx, Vradial &solution) {
  int N = _g.N();
  solution.resize(N);
  solution[0] = std::pow(_g(0), l[idx]+0.5);
  solution[1] = std::pow(_g(1), l[idx]+0.5);
  if ((_o[idx]->n() - _o[idx]->l() - 1) % 2 == 1) {
    solution[0] *= -1;
    solution[1] *= -1;
  }

  for (int k = 1; k < N; ++k) {
    solution[k+1] = ((12 - f[idx][k]*10)*solution[k] - f[idx][k-1]*solution[k-1] + s[idx][k])/f[idx][k+1];
    //std::cout << "outward " << solution[k+1] << std::endl;
  }
}

void IterativeStandardSolver::match(int k, Vradial &o, Vradial &inward, Vradial &outward) {
  o.resize(_g.N());

  ldouble ratio = outward[icl[k]]/inward[icl[k]];
  for (int i = 0; i < _g.N(); ++i) {
    if (i < icl[k]) {
      o[i] = outward[i];
    } else {
      o[i] = ratio*inward[i];
    }
  }

  // normalise it
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
  //std::cout << "norm " << norm << std::endl;
  norm = 1.0/std::sqrt(norm);
  for (int i = 0; i < _g.N(); ++i) {
    o[i] *= norm;
  }
}
