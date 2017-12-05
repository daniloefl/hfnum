#ifndef ITERATIVERENORMALISEDSOLVER_H
#define ITERATIVERENORMALISEDSOLVER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include <vector>
#include "OrbitalMapper.h"

class IterativeRenormalisedSolver {
  public:
    IterativeRenormalisedSolver(const Grid &g, std::vector<Orbital> &o, std::vector<int> &i, OrbitalMapper &om);
    virtual ~IterativeRenormalisedSolver();

    ldouble solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fmn, std::vector<MatrixXld> &Kmn, std::vector<VectorXld> &matched);
    void solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R);
    void solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R);
    void match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward);

    void setFirst();

  private:
    const Grid &_g;
    std::vector<Orbital> &_o;
    std::vector<int> &icl;
    OrbitalMapper &_om;

    int kl;
    bool first;
    ldouble shiftF;
};

#endif

