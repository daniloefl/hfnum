#ifndef LINEARSYSTEMBUILDER_H
#define LINEARSYSTEMBUILDER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"

class LinearSystemBuilder {
  public:
    LinearSystemBuilder();
    virtual ~LinearSystemBuilder();

    void prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<Orbital> &o, std::vector<ldouble> &pot, std::map<int, Vd> &vd, std::map<std::pair<int, int>, Vex> &vex, const Grid &g);
    void propagate(VectorXld &b, std::vector<Orbital> &o, std::vector<ldouble> &dE, const Grid &g, const ldouble gamma);

  private:
};

#endif

