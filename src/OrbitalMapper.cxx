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
  return (index(_o.size(), _o[_o.size()-1].L(), _o[_o.size()-1].L())+1)*_g.N();
}

int OrbitalMapper::index(int k, int l, int m) {
  int idx = 0;
  for (int ki = 0; ki < k; ++ki) {
    for (int li = 0; li < _o[ki].L()+1; ++li) {
      for (int mi = -li; mi < li+1; ++mi) {
        idx++;
      }
    }
  }
  for (int li = 0; li < l; ++li) {
    for (int mi = -li; mi < li+1; ++mi) {
      idx++;
    }
  }
  for (int mi = -l; mi < m; ++mi) {
    idx++;
  }
  return idx;
}

int OrbitalMapper::N() {
  int idx = 0;
  for (int ki = 0; ki < _o.size(); ++ki) {
    for (int li = 0; li < _o[ki].L()+1; ++li) {
      for (int mi = -li; mi < li+1; ++mi) {
        idx++;
      }
    }
  }
  return idx;
}

int OrbitalMapper::orbital(int i) {
  int idx = 0;
  for (int ki = 0; ki < _o.size(); ++ki) {
    for (int li = 0; li < _o[ki].L()+1; ++li) {
      for (int mi = -li; mi < li+1; ++mi) {
        if (i == idx) return ki;
        idx++;
      }
    }
  }
  return -1;
}

// get quantum number l from general index
int OrbitalMapper::l(int i) {
  int idx = 0;
  for (int ki = 0; ki < _o.size(); ++ki) {
    for (int li = 0; li < _o[ki].L()+1; ++li) {
      for (int mi = -li; mi < li+1; ++mi) {
        if (i == idx) return li;
        idx++;
      }
    }
  }
  return -1;
}

// get quantum number m from general index
int OrbitalMapper::m(int i) {
  int idx = 0;
  for (int ki = 0; ki < _o.size(); ++ki) {
    for (int li = 0; li < _o[ki].L()+1; ++li) {
      for (int mi = -li; mi < li+1; ++mi) {
        if (i == idx) return mi;
        idx++;
      }
    }
  }
  return -1;
}
