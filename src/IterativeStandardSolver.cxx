#include "IterativeStandardSolver.h"

#include <iostream>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

IterativeStandardSolver::IterativeStandardSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om, ldouble Z)
  : _g(g), _o(o), icl(i), _om(om), _Z(Z) {

}

IterativeStandardSolver::~IterativeStandardSolver() {
}

void IterativeStandardSolver::setZ(ldouble Z) {
  _Z = Z;
}


VectorXld IterativeStandardSolver::solve(std::vector<ldouble> &E, Vradial &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched) {
  int M = _om.N();

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
        f[idx][k] = 1.0L + std::pow(_g.dx(), 2)/12.0L*(2*std::pow(_g(k), 2)*(E[idx] - pot[k] - vd[idx][k] + vex[std::pair<int,int>(idx,idx)][k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
      }
    } else {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1.0L + std::pow(_g.dx(), 2)/12.0L*((E[idx] - pot[k] - vd[idx][k] + vex[std::pair<int,int>(idx,idx)][k]) - std::pow(_o[idx]->l() + 0.5, 2));
      }
    }
  }

  _i0.resize(M);
  _i1.resize(M);
  ldouble a0 = 1.0;
  for (int k = 0; k < M; ++k) {
    ldouble n = _o[k]->n();
    int l = _o[k]->l();
    ldouble Zeff0 = _Z - _g(0)*(vd[0][0] - vex[std::pair<int,int>(k,k)][0]);
    ldouble Zeff1 = _Z - _g(1)*(vd[0][1] - vex[std::pair<int,int>(k,k)][1]);
    _i0[k] = 2*std::pow(Zeff0/(n*a0), 1.5)*std::pow(_g(0), l)*std::exp(-Zeff0*_g(0)/(n*a0));
    _i1[k] = 2*std::pow(Zeff1/(n*a0), 1.5)*std::pow(_g(1), l)*std::exp(-Zeff1*_g(1)/(n*a0));
    if (l%2 == 1) {
      _i0[k] *= -1;
      _i1[k] *= -1;
    }
  }

  VectorXld F(M);
  F.setZero();

  for (int nIter = 0; nIter < 5; ++nIter) {

    // solve in direct order
    for (int idx = 0; idx < M; ++idx) {
      solveOutward(E, matched, idx, outward[idx]);
      solveInward(E, matched, idx, inward[idx]);
      match(idx, matched[idx], inward[idx], outward[idx]);

      // recalculate non-homogeneus term
      for (int idx1 = 0; idx1 < M; ++idx1) {
        std::fill(s[idx1].begin(), s[idx1].end(), 0);
        for (int idx2 = 0; idx2 < M; ++idx2) {
          if (idx1 == idx2) continue;
          for (int k = 0; k < _g.N(); ++k) {
            if (_g.isLog()) {
              s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
              if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                int lidx = lambdaMap[100*idx1 + idx2];
                s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*lambda[lidx]*matched[idx2][k];
              }
            } else {
              s[idx1][k] += std::pow(_g.dx(), 2)/12.0*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
              if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                int lidx = lambdaMap[100*idx1 + idx2];
                s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*lambda[lidx]*matched[idx2][k];
              }
            }
          }
        }
      } // recalculate non-homogeneous term
    
      // solve in inverse order from current index back to zero
      for (int idxI = idx-1; idxI >= 0; --idxI) {
        solveOutward(E, matched, idxI, outward[idxI]);
        solveInward(E, matched, idxI, inward[idxI]);
        match(idxI, matched[idxI], inward[idxI], outward[idxI]);

        // recalculate non-homogeneus term
        for (int idx1 = 0; idx1 < M; ++idx1) {
          std::fill(s[idx1].begin(), s[idx1].end(), 0);
          for (int idx2 = 0; idx2 < M; ++idx2) {
            if (idx1 == idx2) continue;
            for (int k = 0; k < _g.N(); ++k) {
              if (_g.isLog()) {
                s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
                if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                  int lidx = lambdaMap[100*idx1 + idx2];
                  s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*lambda[lidx]*matched[idx2][k];
                }
              } else {
                s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*vex[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
                if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                  int lidx = lambdaMap[100*idx1 + idx2];
                  s[idx1][k] += std::pow(_g.dx(), 2)/12.0L*lambda[lidx]*matched[idx2][k];
                }
              }
            }
          }
        } // recalculate non-homogeneous term
    
      } // solving in inverse order
    } // solving it in the direct order
    
  }

  for (int idx = 0; idx < M; ++idx) {
    F(idx) = ( (12.0L - 10.0L*f[idx][icl[idx]])*matched[idx][icl[idx]] 
                    - f[idx][icl[idx]-1]*matched[idx][icl[idx]-1]
                    - f[idx][icl[idx]+1]*matched[idx][icl[idx]+1]
                    - s[idx][icl[idx]-1]
                    - s[idx][icl[idx]]
                    - s[idx][icl[idx]+1]
         );
  }

  // alternative to the criteria above:
  //VectorXld W(M);
  //W.setZero();

  //for (int idx = 0; idx < M; ++idx) {
  //  W(idx) = (inward[idx][icl[idx]+1] - inward[idx][icl[idx]-1])*outward[idx][icl[idx]]/(2*_g.dx());
  //  W(idx) -= (outward[idx][icl[idx]+1] - outward[idx][icl[idx]-1])*inward[idx][icl[idx]]/(2*_g.dx());
  //}

  return F;
}

VectorXld IterativeStandardSolver::solve(std::vector<ldouble> &E, Vradial &pot, Vradial &vup, Vradial &vdw, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched) {
  int M = _om.N();

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

  for (int idx1 = 0; idx1 < M; ++idx1) {
    std::fill(s[idx1].begin(), s[idx1].end(), 0);
  }

  for (int idx = 0; idx < M; ++idx) {
    solveOutward(E, matched, idx, outward[idx]);
    solveInward(E, matched, idx, inward[idx]);
    match(idx, matched[idx], inward[idx], outward[idx]);
  }

  VectorXld F(M);
  // calculate first derivative in icl[idx]
  for (int idx = 0; idx < M; ++idx) {
    F(idx) = ( (12.0L - 10.0L*f[idx][icl[idx]])*matched[idx][icl[idx]] 
                    - f[idx][icl[idx]-1]*matched[idx][icl[idx]-1]
                    - f[idx][icl[idx]+1]*matched[idx][icl[idx]+1]
                  );
  }

  return F;
}



void IterativeStandardSolver::solveInward(std::vector<ldouble> &E, std::map<int, Vradial> &matched, int idx, Vradial &solution) {
  int N = _g.N();
  solution.resize(N);
  solution[N-1] = std::pow(_g(N-1), 0.5)*std::exp(-_g(N-1)*std::sqrt(2*std::fabs(E[idx]))/_Z);
  solution[N-2] = std::pow(_g(N-2), 0.5)*std::exp(-_g(N-2)*std::sqrt(2*std::fabs(E[idx]))/_Z);
  for (int k = N-2; k >= 1; --k) {
    solution[k-1] = ((12.0L - f[idx][k]*10.0L)*solution[k] - f[idx][k+1]*solution[k+1] - s[idx][k-1] - s[idx][k] - s[idx][k+1])/f[idx][k-1];
  }
}

void IterativeStandardSolver::solveOutward(std::vector<ldouble> &E, std::map<int, Vradial> &matched, int idx, Vradial &solution) {
  int N = _g.N();
  solution.resize(N);
  ldouble a0 = std::sqrt(2*std::fabs(E[idx]))/_Z;
  solution[0] = std::exp(-_g(0)/_o[idx]->n())*std::pow(_g(0), _o[idx]->l() + 0.5);
  solution[1] = std::exp(-_g(1)/_o[idx]->n())*std::pow(_g(1), _o[idx]->l() + 0.5);
  //if ((_o[idx]->n() - _o[idx]->l() - 1) % 2 == 1) {
  //  solution[0] *= -1;
  //  solution[1] *= -1;
  //}
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
  //solution[0] = std::pow(_g(0), 0.5)*std::pow(_g(0), _o[idx]->l() + 1)*(1 - _Z*_g(0)/(_o[idx]->l() + 1));
  //solution[1] = std::pow(_g(1), 0.5)*std::pow(_g(1), _o[idx]->l() + 1)*(1 - _Z*_g(1)/(_o[idx]->l() + 1));
  for (int k = 1; k < N-2; ++k) {
    solution[k+1] = ((12.0L - f[idx][k]*10.0L)*solution[k] - f[idx][k-1]*solution[k-1] - s[idx][k-1] - s[idx][k] - s[idx][k+1])/f[idx][k+1];
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

  ldouble norm = 0;
  for (int i = 0; i < _g.N(); ++i) {
    ldouble r = _g(i);
    ldouble dr = _g.dx();
    if (_g.isLog()) dr *= r;
    //if (i < _g.N()-1) dr = std::fabs(_g(i+1) - _g(i));
    ldouble ov = o[i];
    if (_g.isLog()) ov *= std::pow(r, -0.5);
    //o_untransformed[i] = ov;
    norm += std::pow(ov*r, 2)*dr;
  }
  for (int i = 0; i < _g.N(); ++i) {
    o[i] /= std::sqrt(norm);
  }

}
