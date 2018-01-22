#ifndef GTO_H
#define GTO_H

#include "Basis.h"
#include "utils.h"

struct GTOUnit {
    ldouble alpha;
    int n;
    int l;
    int m;
};

class GTO : public Basis {
  public:

    GTO();
    virtual ~GTO();

    void load(const std::string &fname);

    void setZ(ldouble Z);

    ldouble dot(int i, int j);
    ldouble T(int i, int j);
    ldouble V(int i, int j);
    ldouble J(int i, int j);
    ldouble K(int i, int j);

    int N();

  protected:

    std::vector<GTOUnit> _u;
    ldouble _Z;

};

#endif
