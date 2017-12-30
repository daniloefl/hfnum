#include "Orbital.h"
#include <vector>
#include "Grid.h"
#include <cmath>

Orbital::Orbital(int s, int initial_n, int initial_l, int initial_m)
 : _N(2), _s(s), _initial_n(initial_n), _initial_l(initial_l), _initial_m(initial_m) {
  _sphHarm.push_back(lm(initial_l, initial_m));
  load();
  _torenorm = true;
}

Orbital::~Orbital() {
  delete [] _wf;
  delete [] _wf_norm;
}

void Orbital::addSphHarm(int l, int m) {
  _sphHarm.push_back(lm(l, m));
  ldouble *wf_new = new ldouble[_N*_sphHarm.size()];
  ldouble *wf_norm_new = new ldouble[_N*_sphHarm.size()];
  for (int idx = 0; idx < _N*(_sphHarm.size()-1); ++idx) {
    wf_new[idx] = _wf[idx];
    wf_norm_new[idx] = _wf_norm[idx];
  }
  for (int idx = _N*(_sphHarm.size()-1); idx < _N*_sphHarm.size(); ++idx) {
    wf_new[idx] = 0;
    wf_norm_new[idx] = 0;
  }
  delete [] _wf;
  delete [] _wf_norm;
  _wf = wf_new;
  _wf_norm = wf_norm_new;
}

const std::vector<lm> &Orbital::getSphHarm() const {
  return _sphHarm;
}

int Orbital::length() const {
  return _sphHarm.size()*_N;
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
  _sphHarm = o._sphHarm;
  load();
  _initial_n = o._initial_n;
  _initial_l = o._initial_l;
  _initial_m = o._initial_m;
  _s = o._s;
  for (int i = 0; i < _sphHarm.size()*_N; ++i) _wf[i] = o._wf[i];
  for (int i = 0; i < _sphHarm.size()*_N; ++i) _wf_norm[i] = o._wf_norm[i];
  _torenorm = true;
}

Orbital &Orbital::operator =(const Orbital &o) {
  _N = o._N;
  delete [] _wf;
  delete [] _wf_norm;
  _sphHarm = o._sphHarm;
  load();
  _initial_n = o._initial_n;
  _initial_l = o._initial_l;
  _initial_m = o._initial_m;
  _s = o._s;
  for (int i = 0; i < _sphHarm.size()*_N; ++i) _wf[i] = o._wf[i];
  for (int i = 0; i < _sphHarm.size()*_N; ++i) _wf_norm[i] = o._wf_norm[i];
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

ldouble &Orbital::operator()(int i, int l, int m) {
  int idx = 0;
  for (; idx < _sphHarm.size(); ++idx) {
    if (_sphHarm[idx].first == l && _sphHarm[idx].second == m) {
      break;
    }
  }
  _torenorm = true;
  return _wf[i + idx*_N];
}

const ldouble Orbital::operator()(int i, int l, int m) const {
  int idx = 0;
  for (; idx < _sphHarm.size(); ++idx) {
    if (_sphHarm[idx].first == l && _sphHarm[idx].second == m) {
      break;
    }
  }
  return _wf[i + idx*_N];
}

void Orbital::normalise(const Grid &g) {
  getNorm(0, initialL(), initialM(), g);
  for (int k = 0; k < _N; ++k) {
    ldouble r = g(k);
    for (int idx = 0; idx < _sphHarm.size(); ++idx) {
      _wf[k + idx*_N] = _wf_norm[k + idx*_N];
      if (g.isLog()) _wf[k + idx*_N] *= std::pow(r, 0.5);
    }
  }
}

python::list Orbital::getNormPython(int lo, int mo) {
  python::list l;
  if (_torenorm) return l;
  int idx = 0;
  for (; idx < _sphHarm.size(); ++idx) {
    if (_sphHarm[idx].first == lo && _sphHarm[idx].second == mo) {
      break;
    }
  }
  for (int k = 0; k < _N; ++k) l.append(_wf_norm[k + idx*_N]);
  return l;
}

python::list Orbital::getCentralNormPython() {
  python::list l;
  if (_torenorm) return l;
  int idx = 0;
  for (; idx < _sphHarm.size(); ++idx) {
    if (_sphHarm[idx].first == initialL() && _sphHarm[idx].second == initialM()) {
      break;
    }
  }
  for (int k = 0; k < _N; ++k) l.append(_wf_norm[k + idx*_N]);
  return l;
}

const ldouble Orbital::getNorm(int i_in, int l_in, int m_in, const Grid &g) {
  if (_torenorm) {
    ldouble norm = 0;
    ldouble integ = 0;
    for (int k = 0; k < _N; ++k) {
      ldouble r = g(k);
      ldouble dr = 0;
      if (k < _N-1) dr = std::fabs(g(k+1) - g(k));
      for (int idx = 0; idx < _sphHarm.size(); ++idx) {
        ldouble ov = _wf[k + idx*_N];
        if (g.isLog()) ov *= std::pow(r, -0.5);
        _wf_norm[k + idx*_N] = ov;
        norm += std::pow(ov*r, 2)*dr;
        integ += ov*r*dr;
      }
    }
    norm = 1.0/std::sqrt(norm);
    //if (integ < 0) norm *= -1;
    for (int k = 0; k < _N; ++k) {
      for (int idx = 0; idx < _sphHarm.size(); ++idx) {
        _wf_norm[k + idx*_N] *= norm;
      }
    }
  }
  _torenorm = false;
  int idx = 0;
  for (; idx < _sphHarm.size(); ++idx) {
    if (_sphHarm[idx].first == l_in && _sphHarm[idx].second == m_in) {
      break;
    }
  }
  return _wf_norm[i_in + idx*_N];
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
  _wf = new ldouble [_sphHarm.size()*_N];
  _wf_norm = new ldouble [_sphHarm.size()*_N];
  for (int idx = 0; idx < _sphHarm.size()*_N; ++idx) {
    _wf[idx] = 0;
    _wf_norm[idx] = 0;
  }
}
