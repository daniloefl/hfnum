#include "RHF.h"
#include "utils.h"
#include "Basis.h"
#include <vector>
#include <iostream>

RHF::RHF() {
  setBasis(&_g);
}

RHF::~RHF() {
}

void RHF::setZ(ldouble Z) {
  _Z = Z;
  _g.setZ(_Z);
}

void RHF::loadBasis(const std::string &fname) {
  _g.load(fname);
}


void RHF::solveRoothan() {
  std::vector<Tr> TLF;
  std::vector<Tr> TLS;

  int N = _g.N();
  _F.resize(N, N);
  _S.resize(N, N);
  _F.setZero();
  _S.setZero();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      //TLS.push_back(S(i, j, _g.dot(i, j)));
      _S(i, j) += _g.dot(i, j);

      _F(i, j) += _g.T(i, j) + _g.V(i, j);
    }
  }
  //_F.setFromTriplets(TLF.begin(), TLF.end());
  //_S.setFromTriplets(TLS.begin(), TLS.end());

  MatrixXld SiF = _S.inverse()*_F;
  EigenSolver<MatrixXld> solver(SiF);
  std::map<ldouble, int> idx;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(solver.eigenvalues()(i).imag()) > 1e-6) continue;
    idx.insert(std::pair<ldouble, int>(-solver.eigenvalues()(i).real(), i));
  }
  for (auto &i : idx) {
    std::cout << "Eigenenergy " << -i.first << ", idx = " << i.second << std::endl;
  }
}

void RHF::solve() {
  solveRoothan();
}

