#include "Grid.h"

#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <Python.h>

#include <cmath>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_lambert.h>

using namespace boost;

Grid::Grid(gridType t, double dx, int N, double rmin, double beta, double Z) {
  _N = N;
  _dx = dx;
  _rmin = rmin;
  _r = new double[_N];
  _t = t;
  _alpha = 0;
  _beta = beta;
  _Z = Z;
  if (_t == expGrid) {
    // x = ln(r) - ln(rmin) = ln(r/rmin), where x = index * dx
    _x0 = std::log(_rmin);
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(_x0 + k*_dx);
  } else if (_t == linExpGrid) {
    // x = a * r + b * ln(r)
    // r = W( (a/b) exp(x/b)) / (a/b)
    // W implemented in gsl_sf_lambert_W0
    double x0 = -_beta*(10.0 + std::log(_Z));
    double rmax = 12.0;
    _alpha = ((x0 + _N*_dx) - _beta*std::log(rmax))/rmax;
    _x0 = x0;
    for (int k = 0; k < _N; ++k) _r[k] = gsl_sf_lambert_W0( (_alpha/_beta) * std::exp((_x0 + k*_dx)/_beta))/(_alpha/_beta);
  } else {
    for (int k = 0; k < _N; ++k) _r[k] = _rmin + k*_dx;
    _x0 = _rmin;
  }
}

Grid::~Grid() {
  delete [] _r;
}

void Grid::reset(gridType t, double dx, int N, double rmin, double beta, double Z) {
  delete [] _r;
  _t = t;
  _N = N;
  _dx = dx;
  _rmin = rmin;
  _r = new double[_N];
  _alpha = 0;
  _beta = beta;
  _Z = Z;
  if (_t == expGrid) {
    _x0 = std::log(_rmin);
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(_x0 + k*_dx);
  } else if (_t == linExpGrid) {
    double x0 = -_beta*(10.0 + std::log(_Z));
    double rmax = 12.0;
    _alpha = ((x0 + _N*_dx) - _beta*std::log(rmax))/rmax;
    _x0 = x0;
    for (int k = 0; k < _N; ++k) _r[k] = gsl_sf_lambert_W0( (_alpha/_beta) * std::exp((_x0 + k*_dx)/_beta))/(_alpha/_beta);
  } else {
    for (int k = 0; k < _N; ++k) _r[k] = _rmin + k*_dx;
    _x0 = _rmin;
  }
}

double Grid::dx() const {
  return _dx;
}
int Grid::N() const {
  return _N;
}

double Grid::operator()(int i) const {
  return _r[i];
}

python::list Grid::getR() const {
  python::list l;
  for (int k = 0; k < N(); ++k) l.append(_r[k]);
  return l;
}

Grid::Grid(const Grid &g) {
  _N = g._N;
  _dx = g._dx;
  _rmin = g._rmin;
  _r = new double[_N];
  _t = g._t;
  if (_t == expGrid) {
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  } else if (_t == linExpGrid) {
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  } else {
    for (int k = 0; k < _N; ++k) _r[k] = _rmin + k*_dx;
  }
  
}

Grid &Grid::operator =(const Grid &g) {
  delete [] _r;
  _N = g._N;
  _dx = g._dx;
  _rmin = g._rmin;
  _r = new double[_N];
  _t = g._t;
  if (_t == expGrid) {
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  } else if (_t == linExpGrid) {
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  } else {
    for (int k = 0; k < _N; ++k) _r[k] = _rmin + k*_dx;
  }
  return *this;
}

bool Grid::isLin() const {
  return _t == linGrid;
}

bool Grid::isLog() const {
  return _t == expGrid;
}

bool Grid::isLinExp() const {
  return _t == linExpGrid;
}

gridType Grid::type() const {
  return _t;
}

