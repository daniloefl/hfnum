#include "OrbitalMapper.h"

#include <vector>
#include "Grid.h"
#include "Orbital.h"

OrbitalMapper::OrbitalMapper(const Grid &g, std::vector<Orbital> &o)
  : _g(g), _o(o) {
}

OrbitalMapper::~OrbitalMapper() {
}


int OrbitalMapper::sparseIndex(int k, int l, int m, int i) {
  return index(k, l, m)*_g.N() + i;
}

int OrbitalMapper::sparseN() {
  int idx = 0;
  for (int ki = 0; ki < _o.size(); ++ki) {
    idx += _o[ki].getSphHarm().size();
  }
  return (idx+1)*_g.N();
}

int OrbitalMapper::index(int k, int l, int m) {
  int idx = 0;
  for (int ki = 0; ki < k; ++ki) {
    idx += _o[ki].getSphHarm().size();
  }
  for (int ii = 0; ii < _o[k].getSphHarm().size(); ++ii) {
    if (_o[k].getSphHarm()[ii] == lm(l, m)) {
      break;
    }
    idx += 1;
  }
  return idx;
}

int OrbitalMapper::N() {
  int idx = 0;
  for (int ki = 0; ki < _o.size(); ++ki) {
    idx += _o[ki].getSphHarm().size();
  }
  return idx;
}

int OrbitalMapper::orbital(int i) {
  int idx = 0;
  int ki = 0;
  int ii = 0;
  for (ki = 0; ki < _o.size(); ++ki) {
    for (ii = 0; ii < _o[ki].getSphHarm().size(); ++ii) {
      if (idx == i) {
        return ki;
      }
      idx += 1;
    }
  }
  return -1;
}

// get quantum number l from general index
int OrbitalMapper::l(int i) {
  int idx = 0;
  int ki = 0;
  int ii = 0;
  for (ki = 0; ki < _o.size(); ++ki) {
    for (ii = 0; ii < _o[ki].getSphHarm().size(); ++ii) {
      if (idx == i) {
        return _o[ki].getSphHarm()[ii].first;
      }
      idx += 1;
    }
  }
  return -1;
}

// get quantum number m from general index
int OrbitalMapper::m(int i) {
  int idx = 0;
  int ki = 0;
  int ii = 0;
  for (ki = 0; ki < _o.size(); ++ki) {
    for (ii = 0; ii < _o[ki].getSphHarm().size(); ++ii) {
      if (idx == i) {
        return _o[ki].getSphHarm()[ii].second;
      }
      idx += 1;
    }
  }
  return -1;
}

