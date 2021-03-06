#include "Orbital.h"
#include <vector>
#include "Grid.h"
#include <cmath>

Orbital::Orbital(int s, int n, int l, int m)
 : _N(2), _s(s), _n(n), _l(l), _m(m), _g(1), _term("") {
  for (int ml = -l; ml <= l; ++ml) {
    for (int ms = -1; ms <= 1; ms += 2) {
      if (_s < 0 && ms < 0 && ml == m)
        _term += '-';
      else if (_s > 0 && ms > 0 && ml == m)
        _term += '+';
      else
        _term += 'N';
    }
  }
  load();
  _torenorm = true;
}

Orbital::Orbital(int n, int l, const std::string term)
 : _N(2), _s(0), _n(n), _l(l), _m(0), _g(0), _term(term) {
  if (_term.size() < 2*(2*l + 1)) {
    for (int k = _term.size(); k < 2*(2*l + 1); ++k) _term += 'N';
  }
  for (int ml_idx = 0; ml_idx < _term.size(); ++ml_idx) {
    int ml = ml_idx/2 - l;
    int ms = 2*(ml_idx % 2) - 1;
    if (_term[ml_idx] == '+' || _term[ml_idx] == '-') _g++;
  }
  
  load();
  _torenorm = true;
}

Orbital::~Orbital() {
  delete [] _wf;
  delete [] _wf_norm;
}

const std::string &Orbital::term() const {
  return _term;
}

int Orbital::length() const {
  return _N;
}

int Orbital::spin() const {
  return _s;
}

void Orbital::spin(int s) {
  _s = s;
}

int Orbital::n() const {
  return _n;
}

int Orbital::l() const {
  return _l;
}

int Orbital::m() const {
  return _m;
}

int Orbital::g() const {
  return _g;
}

Orbital::Orbital(const Orbital &o) {
  _N = o._N;
  load();
  _n = o._n;
  _l = o._l;
  _m = o._m;
  _E = o._E;
  _s = o._s;
  _g = o._g;
  _term = o._term;
  for (int i = 0; i < _N; ++i) _wf[i] = o._wf[i];
  for (int i = 0; i < _N; ++i) _wf_norm[i] = o._wf_norm[i];
  _torenorm = true;
}

Orbital &Orbital::operator =(const Orbital &o) {
  _N = o._N;
  delete [] _wf;
  delete [] _wf_norm;
  load();
  _n = o._n;
  _l = o._l;
  _m = o._m;
  _s = o._s;
  _g = o._g;
  _term = o._term;
  for (int i = 0; i < _N; ++i) _wf[i] = o._wf[i];
  for (int i = 0; i < _N; ++i) _wf_norm[i] = o._wf_norm[i];
  _torenorm = true;
  return *this;
}

void Orbital::N(int N) {
  _N = N;
  delete [] _wf;
  delete [] _wf_norm;
  load();
  _torenorm = true;
}

int Orbital::N() const { return _N; }

ldouble &Orbital::operator()(int i) {
  _torenorm = true;
  return _wf[i];
}

const ldouble Orbital::operator()(int i) const {
  return _wf[i];
}

void Orbital::normalise(const Grid &g) {
  getNorm(0, g);
  for (int k = 0; k < _N; ++k) {
    ldouble r = g(k);
    _wf[k] = _wf_norm[k];
    if (g.isLog()) _wf[k] *= std::pow(r, 0.5);
  }
}

python::list Orbital::getNormPython() {
  python::list l;
  if (_torenorm) return l;
  for (int k = 0; k < _N; ++k) l.append(_wf_norm[k]);
  return l;
}

python::list Orbital::getCentralNormPython() {
  python::list l;
  if (_torenorm) return l;
  for (int k = 0; k < _N; ++k) l.append(_wf_norm[k]);
  return l;
}

const ldouble Orbital::getNorm(int i_in, const Grid &g) {
  if (_torenorm) {
    ldouble norm = 0;
    ldouble integ = 0;
    for (int k = 0; k < _N; ++k) {
      ldouble r = g(k);
      ldouble ov = _wf[k];
      if (g.isLog()) {
        // y = psi sqrt(r), dr = r dx, so psi1 psi2 r^2 dr = y1 y2 / r r^2 r dx = y1 y2 r^2 dx
        _wf_norm[k] = ov*std::pow(r, -0.5);
        norm += std::pow(ov*r, 2)*g.dx();
        integ += ov*std::pow(r, -0.5+2)*g.dx(); // psi r dr = y /sqrt(r) r r dx = y r^(2-0.5) dx
      } else if (g.isLin()) {
        // y = psi , dr = dx, so psi1 psi2 r^2 dr = y1 y2 r^2 dx
        _wf_norm[k] = ov;
        norm += std::pow(ov*r, 2)*g.dx();
        integ += ov*r*g.dx();
      }
    }
    norm = 1.0/std::sqrt(norm);
    //if (integ < 0) norm *= -1;
    for (int k = 0; k < _N; ++k) {
      _wf_norm[k] *= norm;
    }
  }
  _torenorm = false;
  return _wf_norm[i_in];
}

void Orbital::E(ldouble E_in) {
  _E = E_in;
}

void Orbital::setEPython(ldouble E_in) {
  _E = E_in;
}

ldouble Orbital::E() const {
  return _E;
}

ldouble Orbital::EPython() const {
  return _E;
}

void Orbital::load() {
  _wf = new ldouble [_N];
  _wf_norm = new ldouble [_N];
  for (int idx = 0; idx < _N; ++idx) {
    _wf[idx] = 0;
    _wf_norm[idx] = 0;
  }
}
