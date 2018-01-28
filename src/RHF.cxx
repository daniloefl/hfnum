#include "RHF.h"
#include "utils.h"
#include "Basis.h"
#include <vector>
#include <iostream>

#include <boost/python/list.hpp>
#include <boost/python.hpp>

RHF::RHF() {
  setBasis(&_g);
  _Nfilled_up = 0;
  _Nfilled_dw = 0;
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

boost::python::list RHF::getOrbital(int no, int s, int l_proj, int m_proj, boost::python::list r) {
  boost::python::list ret;
  // coefficients in _c_up/dw(:, no)
  MatrixXld c;
  c.resize(_g.N(), 1);
  std::cout << "Getting coefs" << std::endl;
  if (s > 0) {
    c = _c_up.block(0, no, _g.N(), 1);
  } else {
    c = _c_dw.block(0, no, _g.N(), 1);
  }
  std::cout << "Got coefs" << std::endl;
  std::cout << c.transpose() << std::endl;
  int N = _g.N();
  for (int i = 0; i < boost::python::len(r); ++i) {
    ldouble val = 0;
    for (int k = 0; k < N; ++k) {
      val += c(k, 0)*_g.value(k, boost::python::extract<double>(r[i]), l_proj, m_proj);
    }
    ret.append(val);
  }
  return ret;
}

void RHF::solveRoothan() {
  int N = _g.N();

  _F_up.resize(N, N);
  _F_dw.resize(N, N);
  _S.resize(N, N);
  _F_up.setZero();
  _F_dw.setZero();
  _S.setZero();

  // eq. for orbital a is:
  // T |a> + V |a> + Vd |a> + Vex |a> = E_a |a>
  // sum_j c_ja (T|j> + V|j> + Vd|j> + Vex|j>) = sum_j c_ja E_a |j>
  // sum_j c_ja (<i|T|j> + <i|V|j> + <i|Vd|j> + <i|Vex|j>) = sum_j c_ja E_a <i|j>
  // sum_j (T_ij + V_ij + Vd_ij + Vex_ij) C_ja = E_a sum_j S_ij C_ja
  // matrix eq., where C_a is the column vector with components for orbital a
  // (T+V+Vd+Vex) C_a = E_a S C_a
  // [ S^-1 (T+V+Vd+Vex) ] C_a = E_a C_a
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      _S(i, j) += _g.dot(i, j);

      _F_up(i, j) += _g.T(i, j) + _g.V(i, j);
      _F_dw(i, j) += _g.T(i, j) + _g.V(i, j);

      /*
      // add direct term
      // <i|Vd|j> = sum_k <i| int |phi_k|^2/r12 dV1 |j> = sum_k <i| <phi_k|1/r_12|phi_k> |j>
      // |phi_k> = sum_a c_ak |a>
      // <i|Vd|j> = sum_kab c_ak c_bk <i2|<a1|1/r12|b1>|j2> 
      // matrix _Vd(a, b) = sum_k c_ak c_bk
      for (int k = 0; k < N; ++k) {
        for (int a = 0; a < N; ++a) {
          for (int b = 0; b < N; ++b) {
            _F_up(i, j) += _old_c_up(a, k)*_old_c_up(b, k)*_g.ABCD(i, a, b, j);
            _F_dw(i, j) += _old_c_dw(a, k)*_old_c_dw(b, k)*_g.ABCD(i, a, b, j);
          }
        }
      }

      // add exchange term
      // <i|Vex|j> = sum_k <i| int phi*_k |j>/r12 dV1 |phi_k> = sum_k <i| <phi_k|1/r_12|j> |phi_k>
      // |phi_k> = sum_a c_ak |a>
      // <i|Vex|j> = sum_kab c_ak c_bk <i2|<a1|1/r12|j1>|b2> 
      for (int k = 0; k < N; ++k) {
        for (int a = 0; a < N; ++a) {
          for (int b = 0; b < N; ++b) {
            _F_up(i, j) += _old_c_up(a,k)*_old_c_up(b, k)*_g.ABCD(i, a, j, b);
            _F_dw(i, j) += _old_c_dw(a,k)*_old_c_dw(b, k)*_g.ABCD(i, a, j, b);
          }
        }
      }*/
    }
  }

  MatrixXld SiF_up = _S.inverse()*_F_up;
  MatrixXld SiF_dw = _S.inverse()*_F_dw;
  EigenSolver<MatrixXld> solver_up(SiF_up);
  EigenSolver<MatrixXld> solver_dw(SiF_dw);
  std::map<ldouble, int> idx_up;
  std::map<ldouble, int> idx_dw;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(solver_up.eigenvalues()(i).imag()) > 1e-6) continue;
    idx_up.insert(std::pair<ldouble, int>(solver_up.eigenvalues()(i).real(), i));
  }
  for (int i = 0; i < N; ++i) {
    if (std::fabs(solver_dw.eigenvalues()(i).imag()) > 1e-6) continue;
    idx_dw.insert(std::pair<ldouble, int>(solver_dw.eigenvalues()(i).real(), i));
  }
  int count = 0;
  for (auto i = idx_up.begin(); i != std::next(idx_up.begin(), _Nfilled_up); ++i) {
    std::cout << "Orbital " << getOrbitalName(count, 1) << ": " << i->first << ", idx = " << i->second << std::endl;
    _o_up[count].E = i->first;
    count++;
  }
  count = 0;
  for (auto i = idx_dw.begin(); i != std::next(idx_dw.begin(), _Nfilled_dw); ++i) {
    std::cout << "Orbital " << getOrbitalName(count, -1) << ": " << i->first << ", idx = " << i->second << std::endl;
    _o_dw[count].E = i->first;
    count++;
  }

  _c_up.resize(N, N);
  _c_up.setZero();
  _c_dw.resize(N, N);
  _c_dw.setZero();
  int idx_c = 0;
  for (auto k = idx_up.begin(); k != std::next(idx_up.begin(), _Nfilled_up); ++k) {
    int idx_val = k->second;
    for (int m = 0; m < N; ++m) {
      _c_up(m, idx_c) = solver_up.eigenvectors()(m, idx_val).real();
    }
    idx_c++;
  }
  idx_c = 0;
  for (auto k = idx_dw.begin(); k != std::next(idx_dw.begin(), _Nfilled_dw); ++k) {
    int idx_val = k->second;
    for (int m = 0; m < N; ++m) {
      _c_dw(m, idx_c) = solver_dw.eigenvectors()(m, idx_val).real();
    }
    idx_c++;
  }
  // normalise coefficients so that, for a given column:
  // psi(k) = sum_i c(i, k) GTO(i)
  // int |psi(k)|^2 r^2 dr dOmega = 1
  // sum_ij c(i, k)*c(j, k) int GTO(i)*GTO(j) r^2 dr dOmega = 1
  // sum_ij c(i, k)*c(j, k) _g.dot(i, j) = 1
  // so calculate N(k) = sum_ij c(i, k)*c(j, k) _g.dot(i, j) and divide c(:, k) by sqrt(N(k))
  MatrixXld N_up, N_dw;
  N_up.resize(1, _Nfilled_up);
  N_dw.resize(1, _Nfilled_up);
  N_up.setZero();
  N_dw.setZero();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < _Nfilled_up; ++k) {
        N_up(0, k) += _c_up(i, k)*_c_up(j, k)*_S(i, j);
      }
      for (int k = 0; k < _Nfilled_dw; ++k) {
        N_dw(0, k) += _c_dw(i, k)*_c_dw(j, k)*_S(i, j);
      }
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < _Nfilled_up; ++k) {
      _c_up(i, k) *= 1.0/std::sqrt(N_up(0, k));
    }
    for (int k = 0; k < _Nfilled_dw; ++k) {
      _c_dw(i, k) *= 1.0/std::sqrt(N_dw(0, k));
    }
  }
  if (_Nfilled_up > 0) {
    std::cout << "Coefficients for up orbital:" << std::endl;
    std::cout << _c_up.block(0,0,N,_Nfilled_up).transpose() << std::endl;
  }
  if (_Nfilled_dw > 0) {
    std::cout << "Coefficients for down orbital:" << std::endl;
    std::cout << _c_dw.block(0,0,N,_Nfilled_dw).transpose() << std::endl;
  }
}

void RHF::solve() {
  int N = _g.N();

  _Nfilled_up = _o_up.size();
  _Nfilled_dw = _o_dw.size();

  _old_c_up.resize(N, N);
  _old_c_up.setZero();
  _old_c_dw.resize(N, N);
  _old_c_dw.setZero();

  solveRoothan();

  _old_c_up = _c_up;
  _old_c_dw = _c_dw;
}

