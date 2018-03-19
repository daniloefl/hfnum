#ifndef GTO_H
#define GTO_H

#include "Basis.h"
#include "utils.h"

struct GTOUnit {
    ldouble xi;
    int l;
    int m;
};

class GTO : public Basis {
  public:

    GTO();
    virtual ~GTO();

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

    std::vector<GTOUnit> _u;
    ldouble _Z;

};

#endif
