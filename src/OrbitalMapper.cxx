#include "OrbitalMapper.h"

#include <vector>
#include "Grid.h"
#include "Orbital.h"
#include <iostream>

OrbitalMapper::OrbitalMapper(const Grid &g, std::vector<Orbital *> &o)
  : _g(g), _o(o) {
}

OrbitalMapper::~OrbitalMapper() {
}


int OrbitalMapper::sparseIndex(int k, int i) {
  return index(k)*_g.N() + i;
}

int OrbitalMapper::sparseN() {
  int idx = _o.size();
  return idx*_g.N();
}

int OrbitalMapper::index(int k) {
  return k;
}

int OrbitalMapper::N() {
  return _o.size();
}

int OrbitalMapper::orbital(int i) {
  return i;
}

// get quantum number l from general index
int OrbitalMapper::l(int i) {
  return _o[i]->l();
}

// get quantum number m from general index
int OrbitalMapper::m(int i) {
  return _o[i]->m();
}

// get quantum number s from general index
int OrbitalMapper::s(int i) {
  return _o[i]->spin();
}

