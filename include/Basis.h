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
    virtual ldouble RPower(int power, int i, int j) = 0;
    virtual ldouble T(int i, int j) = 0;
    virtual ldouble V(int i, int j) = 0;
    virtual ldouble ABCD(int a, int b, int c, int d) = 0;

    virtual int N() = 0;

    virtual ldouble value(int k, ldouble r, int l_proj, int m_proj) = 0;
    
};

#endif

