#ifndef STO_H
#define STO_H

#include "Basis.h"
#include "utils.h"

// R(r) = 1/sqrt((2*n)!) (2*xi)^(n+0.5) r^(n-1) exp(-xi r)
// Y_lm(theta, phi)
struct STOUnit {
    ldouble xi;
    int n;
    int l;
    int m;
};

class STO : public Basis {
  public:

    STO();
    virtual ~STO();

    void load(const std::string &fname);

    void setZ(ldouble Z);

    ldouble norm(int i);
    ldouble dot(int i, int j);
    ldouble T(int i, int j);
    ldouble V(int i, int j);
    ldouble ABCD(int a, int b, int c, int d);

    int N();

    ldouble value(int k, ldouble r, int l_proj, int m_proj);

  protected:

    std::vector<STOUnit> _u;
    ldouble _Z;

};

#endif
