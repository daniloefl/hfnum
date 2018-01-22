#ifndef BASIS_H
#define BASIS_H

#include <string>
#include "utils.h"

class Basis {
  public:
    Basis();

    virtual ~Basis();

    virtual void load(const std::string &fname) = 0;

    virtual ldouble dot(int i, int j) = 0;
    virtual ldouble T(int i, int j) = 0;
    virtual ldouble V(int i, int j) = 0;
    virtual ldouble J(int i, int j) = 0;
    virtual ldouble K(int i, int j) = 0;

    virtual int N() = 0;
};

#endif

