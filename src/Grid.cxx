#include "Grid.h"

#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <Python.h>

#include <cmath>

using namespace boost;

Grid::Grid(bool isLog, double dx, int N, double rmin) {
  _isLog = isLog;
  _N = N;
  _dx = dx;
  _rmin = rmin;
  _r = new double[_N];
  if (_isLog) {
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  } else {
    for (int k = 0; k < _N; ++k) _r[k] = _rmin + k*_dx;
  }
}

Grid::~Grid() {
  delete [] _r;
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
  _isLog = g._isLog;
  if (_isLog) {
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
  _isLog = g._isLog;
  if (_isLog) {
    for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  } else {
    for (int k = 0; k < _N; ++k) _r[k] = _rmin + k*_dx;
  }
  return *this;
}

bool Grid::isLog() const {
  return _isLog;
}
