#include "HFPy.h"

#include <cstdlib>
#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <sstream>
#include <string>
#include <iomanip>

#include <Python.h>
using namespace boost;

HFPy::HFPy(bool isLog, double dx, int N, double rmin, double Z)
  : _g(isLog, dx, N, rmin), _h(_g, Z) {
  Py_Initialize();
}

HFPy::~HFPy() {
}

void HFPy::gammaSCF(double g) {
  _h.gammaSCF(g);
}

void HFPy::method(int m) {
  _h.method(m);
}

python::list HFPy::getNucleusPotential() {
  python::list l;
  std::vector<ldouble> v = _h.getNucleusPotential();
  for (int k = 0; k < _g.N(); ++k) l.append(v[k]);
  return l;
}
python::list HFPy::getDirectPotential(int k) {
  python::list l;
  std::vector<ldouble> v = _h.getDirectPotential(k);
  for (int k = 0; k < _g.N(); ++k) l.append(v[k]);
  return l;
}

python::list HFPy::getExchangePotential(int k, int k2) {
  python::list l;
  std::vector<ldouble> v = _h.getExchangePotential(k, k2);
  for (int k = 0; k < _g.N(); ++k) l.append(v[k]);
  return l;
}

void HFPy::solve(int NiterSCF, int Niter, double F0stop) {
  _h.solve(NiterSCF, Niter, F0stop);
}

void HFPy::addOrbital(int L, int s, int initial_n, int initial_l, int initial_m) {
  _h.addOrbital(L, s, initial_n, initial_l, initial_m);
}

python::list HFPy::getR() const {
  python::list l;
  for (int k = 0; k < _g.N(); ++k) l.append(_g(k));
  return l;
}

python::list HFPy::getOrbital(int no, int lo, int mo) {
  std::vector<ldouble> o = _h.getOrbital(no, lo, mo);
  python::list l;
  for (int k = 0; k < o.size(); ++k) l.append(o[k]);
  return l;
}

