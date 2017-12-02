#ifndef ITERATIVEGORDONSOLVER_H
#define ITERATIVEGORDONSOLVER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include <vector>

class IterativeGordonSolver {
  public:
    IterativeGordonSolver(const Grid &g, std::vector<Orbital> &o, std::vector<int> &i);
    virtual ~IterativeGordonSolver();

    ldouble solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fmn, std::vector<MatrixXld> &Kmn, std::vector<VectorXld> &matched);
    void solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, int k_init);
    void solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, int k_init);
    void match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward);

  private:
    const Grid &_g;
    std::vector<Orbital> &_o;
    std::vector<int> &icl;
};

#endif

