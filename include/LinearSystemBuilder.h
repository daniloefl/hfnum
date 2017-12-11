#ifndef LINEARSYSTEMBUILDER_H
#define LINEARSYSTEMBUILDER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

class LinearSystemBuilder {
  public:
    LinearSystemBuilder(const Grid &g, std::vector<Orbital> &o, std::vector<int> &i, OrbitalMapper &om);
    virtual ~LinearSystemBuilder();

    void prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<ldouble> &pot, std::map<int, Vd> &vd, std::map<std::pair<int, int>, Vex> &vex);
    void propagate(VectorXld &b, std::vector<ldouble> &dE, const ldouble gamma);

  private:
    const Grid &_g;
    std::vector<Orbital> &_o;
    std::vector<int> &icl;
    OrbitalMapper &_om;
};

#endif

