#include "Orbital.h"
#include <vector>
#include "Grid.h"
#include <cmath>

Orbital::Orbital(int N, int s, int L, int initial_n, int initial_l, int initial_m)
 : _N(N), _s(s), _L(L), _initial_n(initial_n), _initial_l(initial_l), _initial_m(initial_m) {
  load();
  _torenorm = true;
}

Orbital::~Orbital() {
  delete [] _wf;
  delete [] _wf_norm;
}

int Orbital::length() const {
  int lsum = 0;
  for (int l = 0; l < _L+1; ++l) {
    lsum += 2*l + 1;
  }
  return lsum*_N;
}

int Orbital::spin() const {
  return _s;
}

void Orbital::spin(int s) {
  _s = s;
}

int Orbital::initialN() const {
  return _initial_n;
}

int Orbital::initialL() const {
  return _initial_l;
}

int Orbital::initialM() const {
  return _initial_m;
}

Orbital::Orbital(const Orbital &o) {
  _N = o._N;
  _L = o._L;
  load();
  int lsum = 0;
  for (int l = 0; l < _L+1; ++l) {
    lsum += 2*l + 1;
  }
  _initial_n = o._initial_n;
  _initial_l = o._initial_l;
  _initial_m = o._initial_m;
  _s = o._s;
  for (int i = 0; i < lsum*_N; ++i) _wf[i] = o._wf[i];
  for (int i = 0; i < lsum*_N; ++i) _wf_norm[i] = o._wf_norm[i];
  _torenorm = true;
}

Orbital &Orbital::operator =(const Orbital &o) {
  _N = o._N;
  _L = o._L;
  delete [] _wf;
  delete [] _wf_norm;
  load();
  int lsum = 0;
  for (int l = 0; l < _L+1; ++l) {
    lsum += 2*l + 1;
  }
  _initial_n = o._initial_n;
  _initial_l = o._initial_l;
  _initial_m = o._initial_m;
  _s = o._s;
  for (int i = 0; i < lsum*_N; ++i) _wf[i] = o._wf[i];
  for (int i = 0; i < lsum*_N; ++i) _wf_norm[i] = o._wf_norm[i];
  _torenorm = true;
  return *this;
}

void Orbital::N(int N) { _N = N; }
void Orbital::L(int L) { _L = L; }

int Orbital::N() const { return _N; }
int Orbital::L() const { return _L; }

double &Orbital::operator()(int i, int l, int m) {
  int lsum = 0;
  for (int li = 0; li < l; ++li) {
    lsum += 2*li + 1;
  }
  int msum = (m+l)*_N;
  _torenorm = true;
  return _wf[i + lsum*_N + msum];
}

const double Orbital::operator()(int i, int l, int m) const {
  int lsum = 0;
  for (int li = 0; li < l; ++li) {
    lsum += 2*li + 1;
  }
  int msum = (m+l)*_N;
  return _wf[i + lsum*_N + msum];
}

const double Orbital::getNorm(int i_in, int l_in, int m_in, const Grid &g) {
  if (_torenorm) {
    double norm = 0;
    for (int k = 0; k < _N; ++k) {
      double r = g(k);
      double dr = 0;
      if (k < _N-1) dr = std::fabs(g(k+1) - g(k));
      for (int l = 0; l < L()+1; ++l) {
        int lsum = 0;
        for (int li = 0; li < l; ++li) {
          lsum += 2*li + 1;
        }
        for (int m = -l; m < l+1; ++m) {
          int msum = (m+l)*_N;
          double ov = _wf[k + lsum*_N + msum]*std::pow(r, -0.5);
          _wf_norm[k + lsum*_N + msum] = ov;
          norm += std::pow(ov*r, 2)*dr;
        }
      }
    }
    for (int k = 0; k < _N; ++k) {
      for (int l = 0; l < L()+1; ++l) {
        int lsum = 0;
        for (int li = 0; li < l; ++li) {
          lsum += 2*li + 1;
        }
        for (int m = -l; m < l+1; ++m) {
          int msum = (m+l)*_N;
          _wf_norm[k + lsum*_N + msum] /= std::sqrt(norm);
        }
      }
    }
  }
  _torenorm = false;
  int lsum = 0;
  for (int li = 0; li < l_in; ++li) {
    lsum += 2*li + 1;
  }
  int msum = (m_in+l_in)*_N;
  return _wf_norm[i_in + lsum*_N + msum];
}

void Orbital::E(double E_in) {
  _E = E_in;
}

double Orbital::E() const {
  return _E;
}

void Orbital::load() {
  int lsum = 0;
  for (int l = 0; l < _L+1; ++l) {
    lsum += 2*l + 1;
  }
  _wf = new double [lsum*_N];
  _wf_norm = new double [lsum*_N];
}
