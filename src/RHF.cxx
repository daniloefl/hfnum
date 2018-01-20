#include "RHF.h"

RHF::RHF() {
  setBasis(&_g);
}

RHF::~RHF() {
}

void RHF::setZ(ldouble Z) {
  _Z = Z;
  _g.setZ(_Z);
}

void RHF::loadBasis(const std::string &fname) {
  _g.load(fname);
}


void RHF::solveRoothan() {
}

void RHF::solve() {
}

