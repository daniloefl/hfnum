#include "IterativeGordonSolver.h"

#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"

IterativeGordonSolver::IterativeGordonSolver(const Grid &g, std::vector<Orbital> &o, std::vector<int> &i)
  : _g(g), _o(o), icl(i) {
}

IterativeGordonSolver::~IterativeGordonSolver() {
}


ldouble IterativeGordonSolver::solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched) {
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += 2*_o[k].L()+1;
  }

  std::vector< std::vector<VectorXld> > inward(_o.size());
  std::vector< std::vector<VectorXld> > outward(_o.size());
  std::vector<VectorXld> fix_inward(_g.N());
  std::vector<VectorXld> fix_outward(_g.N());
  matched.resize(_g.N());

  for (int k = 0; k < _o.size(); ++k) {
    inward[k] = std::vector<VectorXld>(_g.N());
    outward[k] = std::vector<VectorXld>(_g.N());
    solveOutward(E, l, outward[k], Fm, Km, k);
    solveInward(E, l, inward[k], Fm, Km, k);
  }
  int idx1 = 0;
  MatrixXld D(2*_o.size(), 2*_o.size());
  MatrixXld Da(_o.size(), _o.size());
  MatrixXld Db(_o.size(), _o.size());
  MatrixXld Dc(_o.size(), _o.size());
  MatrixXld Dd(_o.size(), _o.size());
  for (int k = 0; k < _o.size(); ++k) {
    int icl0 = icl[k];
    ldouble dr1 = _g(icl0) - _g(icl0-1);
    ldouble dr2 = _g(icl0+1) - _g(icl0);
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        if (l == _o[k].initialL() && m == _o[k].initialM()) {
          for (int k_init = 0; k_init < _o.size(); ++k_init) {
            D(idx1, k_init) = outward[k_init][icl0](idx1);
            Da(idx1, k_init) = outward[k_init][icl0](idx1);

            D(idx1, _o.size() + k_init) = inward[k_init][icl0](idx1);
            Db(idx1, k_init) = inward[k_init][icl0](idx1);

            D(_o.size()+idx1, k_init) = outward[k_init][icl0+1](idx1);
            Dc(idx1, k_init) = outward[k_init][icl0+1](idx1);

            D(_o.size()+idx1, _o.size() + k_init) = inward[k_init][icl0+1](idx1);
            Dd(idx1, k_init) = inward[k_init][icl0+1](idx1);
          }
        }
        idx1 += 1;
      }
    }
  }
  ldouble F = D.determinant();
  VectorXld left(_o.size());
  for (int i = 0; i < _o.size(); ++i) left(i) = 1.0;
  VectorXld r(_o.size());
  r = (Db+Dd).inverse()*((Da+Dc)*left);

  for (int i = 0; i < _g.N(); ++i) {
    fix_inward[i].resize(M);
    fix_outward[i].resize(M);
    fix_inward[i].setZero();
    fix_outward[i].setZero();
  }

  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 0; i < _g.N(); ++i) {
          if (i <= icl[k]+1) {
            for (int ks = 0; ks < _o.size(); ++ks) {
              fix_outward[i](idx) += outward[ks][i](idx)*left(ks);
            }
          }
          if (i >= icl[k]-1) {
            for (int ks = 0; ks < _o.size(); ++ks) {
              fix_inward[i](idx) += inward[ks][i](idx)*r(ks);
            }
          }
        }
        idx += 1;
      }
    }
  }
  match(matched, fix_inward, fix_outward);

  return std::pow(F, 2);
}



void IterativeGordonSolver::solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, int k_init) {
  int N = _g.N();
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += 2*_o[k].L()+1;
  }
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
  }
  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        if (l == _o[k].initialL() && m == _o[k].initialM()) {
          solution[N-1](idx) = 0; //std::exp(-std::sqrt(2*std::fabs(E[k]))*_g(N-1));
          solution[N-2](idx) = 1; //std::exp(-std::sqrt(2*std::fabs(E[k]))*_g(N-2));
          if (k == k_init) solution[N-2](idx) *= 2;
        }
        idx += 1;
      }
    }
  }
  idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = N-2; i >= icl[k]-1; --i) {
          //JacobiSVD<MatrixXld> dec(Fm[i-1], ComputeThinU | ComputeThinV);
          //solution[i-1] = dec.solve((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1]));
          solution[i-1] = Km[i-1]*((MatrixXld::Identity(M,M)*12 - (Fm[i])*10)*solution[i] - (Fm[i+1]*solution[i+1])); 
        }
        idx += 1;
      }
    }
  }
}

void IterativeGordonSolver::solveOutward(std::vector<ldouble> &E, std::vector<int> &li, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, int k_init) {
  int N = _g.N();
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += 2*_o[k].L()+1;
  }
  for (int i = 0; i < N; ++i) {
    solution[i].resize(M);
  }
  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        if (l == _o[k].initialL() && m == _o[k].initialM()) {
          if (_g.isLog()) {
            solution[0](idx) = 0;//std::pow(_Z*_g(0)/((ldouble) _o[k].initialN()), li[k]+0.5);
            solution[1](idx) = 1; //std::pow(_Z*_g(1)/((ldouble) _o[k].initialN()), li[k]+0.5);
          } else {
            solution[0](idx) = 0;
            solution[1](idx) = 1;//std::pow(_Z*_g(1)/((ldouble) _o[k].initialN()), li[k]+1);
          }
          if ((_o[k].initialN() - _o[k].initialL() - 1) % 2 == 1) {
            solution[0](idx) *= -1;
            solution[1](idx) *= -1;
          }
          if (k == k_init) solution[1](idx) *= 2;
        }
        idx += 1;
      }
    }
  }
  idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 1; i <= icl[k]+1; ++i) {
          //JacobiSVD<MatrixXld> dec(Fm[i+1], ComputeThinU | ComputeThinV);
          //solution[i+1] = dec.solve((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
          solution[i+1] = Km[i+1]*((MatrixXld::Identity(M, M)*12 - (Fm[i])*10)*solution[i] - (Fm[i-1]*solution[i-1]));
        }
        idx += 1;
      }
    }
  }
}
void IterativeGordonSolver::match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward) {
  int M = 0;
  for (int k = 0; k < _o.size(); ++k) {
    M += 2*_o[k].L()+1;
  }
  for (int i = 0; i < _g.N(); ++i) {
    o[i].resize(M);
  }

  int idx = 0;
  for (int k = 0; k < _o.size(); ++k) {
    ldouble ratio = outward[icl[k]](k)/inward[icl[k]](k);
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        for (int i = 0; i < _g.N(); ++i) {
          if (i < icl[k]) {
            o[i](idx) = outward[i](idx);
          } else {
            o[i](idx) = ratio*inward[i](idx);
          }
        }
        idx += 1;
      }
    }
  }
}
