#include "Grid.h"

#include <cmath>

Grid::Grid(double dx, int N, double rmin) {
  _N = N;
  _dx = dx;
  _rmin = rmin;
  _r = new double[_N];
  for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
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

Grid::Grid(const Grid &g) {
  _N = g._N;
  _dx = g._dx;
  _rmin = g._rmin;
  _r = new double[_N];
  for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  
}

Grid &Grid::operator =(const Grid &g) {
  delete [] _r;
  _N = g._N;
  _dx = g._dx;
  _rmin = g._rmin;
  _r = new double[_N];
  for (int k = 0; k < _N; ++k) _r[k] = std::exp(std::log(_rmin) + k*_dx);
  return *this;
}
