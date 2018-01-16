#include "PerturbativeCorrection.h"

using namespace boost;

PerturbativeCorrection::PerturbativeCorrection()
  : _g(new Grid(true, 1e-1, 10, 1e-3)), _Z(1) {
}

PerturbativeCorrection::~PerturbativeCorrection() {
  for (auto &o : _o) {
    delete o;
  }
  _o.clear();
  if (_g) delete _g;
}

ldouble PerturbativeCorrection::Z() {
  return _Z;
}

std::vector<ldouble> PerturbativeCorrection::getCorrectedE() {
  std::vector<ldouble> o;
  for (int i = 0; i < _Ec.size(); ++i) o.push_back(_Ec[i]+_o[i]->E());
  return o;
}

boost::python::list PerturbativeCorrection::getCorrectedEPython() {
  python::list l;
  for (int i = 0; i < _Ec.size(); ++i) l.append(_Ec[i]+_o[i]->E());
  return l;
}

boost::python::list PerturbativeCorrection::getR() {
  return _g->getR();
}

MatrixXcld PerturbativeCorrection::getCoefficients() {
  return _c;
}

boost::python::list PerturbativeCorrection::getCoefficientsPython() {
  int N = _o.size();
  python::list data;
  for (int i = 0; i < N; ++i) {
    python::list row;
    for (int j = 0; j < N; ++j) {
      row.append(_c(i, j));
    }
    data.append(row);
  }
  return data;
}

std::vector<ldouble> PerturbativeCorrection::getOrbitalCentral(int no) {
  Orbital *o = _o[no];
  std::vector<ldouble> res;
  for (int k = 0; k < _g->N(); ++k) {
    res.push_back(o->getNorm(k, o->initialL(), o->initialM(), *_g));
  }
  return res;
}

python::list PerturbativeCorrection::getOrbitalCentralPython(int no) {
  python::list l;
  std::vector<ldouble> v = getOrbitalCentral(no);
  for (int k = 0; k < _g->N(); ++k) l.append(v[k]);
  return l;
}

int PerturbativeCorrection::getNOrbitals() {
  return _o.size();
}

int PerturbativeCorrection::getOrbital_n(int no) {
  return _o[no]->initialN();
}

std::string PerturbativeCorrection::getOrbitalName(int no) {
  std::string name = "";
  name += std::to_string(_o[no]->initialN());
  int l = _o[no]->initialL();
  int m = _o[no]->initialM();
  int s = _o[no]->spin();
  if (l == 0) name += "s";
  else if (l == 1) name += "p";
  else if (l == 2) name += "d";
  else if (l == 3) name += "f";
  else if (l == 4) name += "g";
  else if (l == 5) name += "h";
  else name += "?";
  name += "_{m=";
  name += std::to_string(m);
  name += "}";
  if (s > 0)
    name += "^+";
  else
    name += "^-";
  return name;
}

ldouble PerturbativeCorrection::getOrbital_E(int no) {
  return _o[no]->E();
}

int PerturbativeCorrection::getOrbital_l(int no) {
  return _o[no]->initialL();
}

int PerturbativeCorrection::getOrbital_m(int no) {
  return _o[no]->initialM();
}

int PerturbativeCorrection::getOrbital_s(int no) {
  return _o[no]->spin();
}
