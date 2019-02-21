#include "IterativeStandardSolver.h"

#include <iostream>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

IterativeStandardSolver::IterativeStandardSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om, ldouble Z)
  : _g(g), _o(o), icl(i), _om(om), _Z(Z) {

  _i0.clear();
  _i1.clear();
}

IterativeStandardSolver::~IterativeStandardSolver() {
}

void IterativeStandardSolver::setZ(ldouble Z) {
  _Z = Z;
}


VectorXld IterativeStandardSolver::solve(VectorXld &E, Vradial &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, VectorXld &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched) {
  int M = _om.N();

  for (int idx = 0; idx < M; ++idx) {
    if (matched.find(idx) == matched.end()) matched.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (inward.find(idx) == inward.end()) inward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (outward.find(idx) == outward.end()) outward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (homoInward.find(idx) == homoInward.end()) homoInward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (homoOutward.find(idx) == homoOutward.end()) homoOutward.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (f.find(idx) == f.end()) f.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (s.find(idx) == s.end()) s.insert(std::pair<int, Vradial>(idx, Vradial()));
    if (snew.find(idx) == snew.end()) snew.insert(std::pair<int, Vradial>(idx, Vradial()));
  }
  for (int idx = 0; idx < M; ++idx) {
    matched[idx].resize(_g.N(), 0);
    inward[idx].resize(_g.N(), 0);
    outward[idx].resize(_g.N(), 0);
    homoInward[idx].resize(_g.N(), 0);
    homoOutward[idx].resize(_g.N(), 0);
    f[idx].resize(_g.N(), 0);
    s[idx].resize(_g.N(), 0);
    snew[idx].resize(_g.N(), 0);
  }

  // get part multiplying function in the equation
  coeff = vd;
  // get non-homogeneous term in the equation
  indep = vex;
  for (int idx = 0; idx < M; ++idx) {
    for (int k = 0; k < _g.N(); ++k) {
      coeff[idx][k] += -indep[std::pair<int, int>(idx, idx)][k];
      coeff[idx][k] += pot[k];
      indep[std::pair<int, int>(idx, idx)][k] = 0;
    }
  }
  for (int idx1 = 0; idx1 < M; ++idx1) {
    double maxIdx1 = 0;
    for (int k = 0; k < _g.N(); ++k) {
      if (std::fabs(matched[idx1][k]) > maxIdx1) maxIdx1 = std::fabs(matched[idx1][k]);
    }

    for (int idx2 = 0; idx2 < M; ++idx2) {
      if (idx1 == idx2) continue;
      for (int k = 0; k < _g.N(); ++k) {
        if (std::fabs(matched[idx1][k]) > maxIdx1*0.05) {
          coeff[idx1][k] += -indep[std::pair<int, int>(idx1, idx2)][k]*matched[idx2][k]/matched[idx1][k];
          indep[std::pair<int, int>(idx1, idx2)][k] = 0;
        }
      }
    }
  }

  for (int idx = 0; idx < M; ++idx) {
    if (_g.isLog()) {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1.0L + std::pow(_g.dx(), 2)/12.0L*(2*std::pow(_g(k), 2)*(E(idx) - coeff[idx][k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
      }
    } else {
      for (int k = 0; k < _g.N(); ++k) {
        f[idx][k] = 1.0L + std::pow(_g.dx(), 2)/12.0L*((E(idx) - coeff[idx][k]) - std::pow(_o[idx]->l() + 0.5, 2));
      }
    }
  }

  ldouble ic = 0.5;

  if (_i0.size() != _o.size()) {
    _i0.resize(_o.size(), 0);
    _i1.resize(_o.size(), 0);
    for (int idx = 0; idx < M; ++idx) {
      ldouble Zeff = _o[idx]->n()*std::sqrt(std::fabs(2*E(idx)));
      _i0[idx] = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(0)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(0)/((ldouble) _o[idx]->n()));
      _i1[idx] = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(1)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(1)/((ldouble) _o[idx]->n()));
    }
  }

  VectorXld F(M);
  F.setZero();

  // Procedure:
  // Start with non-homogeneous terms s for each equation at zero (they come from the exchange potential multiplying solutions for other orbitals)
  // Do the following 3 times
  //   -> For current eq. k from 0 to M:
  //     -> Solve eq. k using last non-homogeneus term s(k) for that eq.
  //     -> Recalculate all the non-homogeneous terms snew(*) with the last solution
  //     -> Use as the new non-homogeneus terms: s <- ic * snew + (1 - ic) * s [where ic is a constant, set at 0.3]
  //     -> For each eq. j from M-1 to 0:
  //       -> Solve the eq. j using the new non-homogeneus term s(j)
  //       -> Recalculate all the non-homogeneous terms snew(*) with the last solution
  //       -> Use as the new non-homogeneus terms: s <- ic * snew + (1 - ic) * s [where ic is a constant, set at 0.3]
  //   -> For current eq. k from 0 to M:
  //     -> Solve eq. k using a *zero* non-homogeneous term.
  //     -> Fix the initial conditions i0 and i1 for the equations based on the ratio of the homogeneous and non-homogeneous solutions
  //         --> See fixIC
  //     -> Recalculate all the non-homogeneous terms snew(*) with the last solution
  //     -> Use as the new non-homogeneus terms: s <- ic * snew + (1 - ic) * s [where ic is a constant, set at 0.3]
  //   -> Go back to the beginning to repeat this.

  for (int nIter = 0; nIter < 3; ++nIter) {

    // solve in direct order
    for (int idx = 0; idx < M; ++idx) {
      solveOutward(E, idx, outward[idx]);
      solveInward(E, idx, inward[idx]);
      match(idx, matched[idx], inward[idx], outward[idx]);

      // recalculate non-homogeneus term
      for (int idx1 = 0; idx1 < M; ++idx1) {
        std::fill(snew[idx1].begin(), snew[idx1].end(), 0);
        for (int idx2 = 0; idx2 < M; ++idx2) {
          if (idx1 == idx2) continue;
          for (int k = 0; k < _g.N(); ++k) {
            if (_g.isLog()) {
              snew[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*indep[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
              if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                int lidx = lambdaMap[100*idx1 + idx2];
                snew[idx1][k] += -std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*lambda(lidx)*matched[idx2][k];
              }
            } else {
              snew[idx1][k] += std::pow(_g.dx(), 2)/12.0*indep[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
              if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                int lidx = lambdaMap[100*idx1 + idx2];
                snew[idx1][k] += -std::pow(_g.dx(), 2)/12.0L*lambda(lidx)*matched[idx2][k];
              }
            }
          }
        }
      } // recalculate non-homogeneous term
      for (int idx1 = 0; idx1 < M; ++idx1) {
        for (int k = 0; k < _g.N(); ++k) {
          s[idx1][k] = snew[idx1][k]*ic + (1-ic)*s[idx1][k];
        }
      }
      
    
      // solve in inverse order from current index back to zero
      for (int idxI = idx-1; idxI >= 0; --idxI) {
        solveOutward(E, idxI, outward[idxI]);
        solveInward(E, idxI, inward[idxI]);
        match(idxI, matched[idxI], inward[idxI], outward[idxI]);

        // recalculate non-homogeneus term
        for (int idx1 = 0; idx1 < M; ++idx1) {
          std::fill(snew[idx1].begin(), snew[idx1].end(), 0);
          for (int idx2 = 0; idx2 < M; ++idx2) {
            if (idx1 == idx2) continue;
            for (int k = 0; k < _g.N(); ++k) {
              if (_g.isLog()) {
                snew[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*indep[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
                if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                  int lidx = lambdaMap[100*idx1 + idx2];
                  snew[idx1][k] += -std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*lambda(lidx)*matched[idx2][k];
                }
              } else {
                snew[idx1][k] += std::pow(_g.dx(), 2)/12.0L*indep[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
                if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
                  int lidx = lambdaMap[100*idx1 + idx2];
                  snew[idx1][k] += -std::pow(_g.dx(), 2)/12.0L*lambda(lidx)*matched[idx2][k];
                }
              }
            }
          }
        } // recalculate non-homogeneous term
        for (int idx1 = 0; idx1 < M; ++idx1) {
          for (int k = 0; k < _g.N(); ++k) {
            s[idx1][k] = snew[idx1][k]*ic + (1-ic)*s[idx1][k];
          }
        }
    
      } // solving in inverse order
    } // solving it in the direct order

    for (int idx = 0; idx < M; ++idx) {
      solveOutward(E, idx, homoOutward[idx], true);
      solveInward(E, idx, homoInward[idx], true);
      ldouble rhicl = homoOutward[idx][icl[idx]]/homoInward[idx][icl[idx]];
      for (int i = 0; i < _g.N(); ++i) {
        homoInward[idx][i] *= rhicl;
      }
    }
    for (int idx = 0; idx < M; ++idx) {
      solveOutward(E, idx, outward[idx]);
      solveInward(E, idx, inward[idx]);
    }

    for (int idx = 0; idx < M; ++idx) {
      fixIC(idx, inward[idx], outward[idx], homoInward[idx], homoOutward[idx]);
      solveInward(E, idx, inward[idx]);
      solveOutward(E, idx, outward[idx]);
      match(idx, matched[idx], inward[idx], outward[idx]);
    }

    // recalculate non-homogeneus term
    for (int idx1 = 0; idx1 < M; ++idx1) {
      std::fill(snew[idx1].begin(), snew[idx1].end(), 0);
      for (int idx2 = 0; idx2 < M; ++idx2) {
        if (idx1 == idx2) continue;
        for (int k = 0; k < _g.N(); ++k) {
          if (_g.isLog()) {
            snew[idx1][k] += std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*indep[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
            if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
              int lidx = lambdaMap[100*idx1 + idx2];
              snew[idx1][k] += -std::pow(_g.dx(), 2)/12.0L*2.0L*std::pow(_g(k), 2)*lambda(lidx)*matched[idx2][k];
            }
          } else {
            snew[idx1][k] += std::pow(_g.dx(), 2)/12.0*indep[std::pair<int,int>(idx1, idx2)][k]*matched[idx2][k];
            if (lambdaMap.find(100*idx1 + idx2) != lambdaMap.end()) {
              int lidx = lambdaMap[100*idx1 + idx2];
              snew[idx1][k] += -std::pow(_g.dx(), 2)/12.0L*lambda(lidx)*matched[idx2][k];
            }
          }
        }
      }
    } // recalculate non-homogeneous term
    for (int idx1 = 0; idx1 < M; ++idx1) {
      for (int k = 0; k < _g.N(); ++k) {
        s[idx1][k] = snew[idx1][k]*ic + (1-ic)*s[idx1][k];
      }
    }
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
  //  W(idx) = (inward[idx][icl[idx]+1] - inward[idx][icl[idx]-1])*outward[idx][icl[idx]]; // /(2*_g.dx()); // no nead to divide by this common factor
  //  W(idx) -= (outward[idx][icl[idx]+1] - outward[idx][icl[idx]-1])*inward[idx][icl[idx]]; // /(2*_g.dx());
  //}

  return F;
}

VectorXld IterativeStandardSolver::solve(VectorXld &E, Vradial &pot, Vradial &vup, Vradial &vdw, VectorXld &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched) {
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
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*(2*std::pow(_g(k), 2)*(E(idx) - pot[k] - vup[k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
        if (_o[idx]->term().find('-') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*(2*std::pow(_g(k), 2)*(E(idx) - pot[k] - vdw[k]) - std::pow(((ldouble ) _o[idx]->l()) + 0.5, 2));
      }
    } else {
      for (int k = 0; k < _g.N(); ++k) {
        if (_o[idx]->term().find('+') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*((E(idx) - pot[k] - vup[k]) - std::pow(_o[idx]->l() + 0.5, 2));
        if (_o[idx]->term().find('-') != std::string::npos)
          f[idx][k] = 1 + std::pow(_g.dx(), 2)/12.0*((E(idx) - pot[k] - vdw[k]) - std::pow(_o[idx]->l() + 0.5, 2));
      }
    }
  }

  for (int idx1 = 0; idx1 < M; ++idx1) {
    std::fill(s[idx1].begin(), s[idx1].end(), 0);
  }

  for (int idx = 0; idx < M; ++idx) {
    solveOutward(E, idx, outward[idx]);
    solveInward(E, idx, inward[idx]);
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



void IterativeStandardSolver::solveInward(VectorXld &E, int idx, Vradial &solution, bool homo) {
  int N = _g.N();
  solution.resize(N);
  ldouble Zeff = _o[idx]->n()*std::sqrt(std::fabs(2*E(idx)));
  solution[N-1] = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(N-1)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(N-1)/((ldouble) _o[idx]->n()));
  solution[N-2] = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(N-2)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(N-2)/((ldouble) _o[idx]->n()));

  if (homo) {
    for (int k = N-2; k >= 1; --k) {
      solution[k-1] = ((12.0L - f[idx][k]*10.0L)*solution[k] - f[idx][k+1]*solution[k+1])/f[idx][k-1];
    }
  } else {
    for (int k = N-2; k >= 1; --k) {
      solution[k-1] = ((12.0L - f[idx][k]*10.0L)*solution[k] - f[idx][k+1]*solution[k+1] - s[idx][k-1] - s[idx][k] - s[idx][k+1])/f[idx][k-1];
    }
  }
}

void IterativeStandardSolver::solveOutward(VectorXld &E, int idx, Vradial &solution, bool homo) {
  int N = _g.N();
  solution.resize(N);
  ldouble Zeff = _o[idx]->n()*std::sqrt(std::fabs(2*E(idx)));
  solution[0] = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(0)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(0)/((ldouble) _o[idx]->n()));
  solution[1] = std::sqrt(Zeff)*2*std::pow(Zeff/((ldouble) _o[idx]->n()), 1.5)*std::pow(_g(1)/((ldouble) _o[idx]->n()), _o[idx]->l() + 0.5)*std::exp(-Zeff*_g(1)/((ldouble) _o[idx]->n()));
  if (idx < _i0.size()) {
    solution[0] = _i0[idx];
    solution[1] = _i1[idx];
  }
  if (homo) {
    for (int k = 1; k < N-2; ++k) {
      solution[k+1] = ((12.0L - f[idx][k]*10.0L)*solution[k] - f[idx][k-1]*solution[k-1])/f[idx][k+1];
    }
  } else {
    for (int k = 1; k < N-2; ++k) {
      solution[k+1] = ((12.0L - f[idx][k]*10.0L)*solution[k] - f[idx][k-1]*solution[k-1] - s[idx][k-1] - s[idx][k] - s[idx][k+1])/f[idx][k+1];
    }
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
    ldouble ov = o[i];
    // in log scale: y = psi sqrt(r) and dr = r dx, so psi1 psi2 r^2 dr = y1 y2 / r r^2 r dx = y1 y2 r^2 dx
    // in lin scale y = psi, dr = dx, so psi1 psi2 r^2 dr = y1 y2 r^2 dx
    if (_g.isLog()) norm += ov*ov*r*r*_g.dx();
    else if (_g.isLin()) norm += ov*ov*r*r*_g.dx();
  }
  for (int i = 0; i < _g.N(); ++i) {
    o[i] /= std::sqrt(norm);
  }

}

void IterativeStandardSolver::fixIC(int idx, Vradial &in, Vradial &out, Vradial &hin, Vradial &hout) {
  if (hout[icl[idx]] - hin[icl[idx]] == 0) return;
  ldouble ricl = out[icl[idx]]/in[icl[idx]];
  for (int i = 0; i < _g.N(); ++i) {
    in[i] *= ricl;
  }

  ldouble alpha = -(out[icl[idx]+1] - in[icl[idx]+1])/(hout[icl[idx]+1] - hin[icl[idx]+1]);
  if (alpha == -1) {
    _i0[idx] = out[0];
    _i1[idx] = out[1];
    return;
  }
  for (int i = 0; i < _g.N(); ++i) {
    in[i] = in[i] + alpha*hin[i];
    out[i] = out[i] + alpha*hout[i];
  }
  Vradial m(in.size());
  match(idx, m, in, out);
  _i0[idx] = m[0];
  _i1[idx] = m[1];
  //std::cout << "alpha = " << alpha << " " << idx << " " << out[icl[idx]+1] << " " << in[icl[idx]+1] << " " << hout[icl[idx]+1] << " " << hin[icl[idx]+1] << " " << _i0[idx] << " " << _i1[idx] << std::endl;
}

