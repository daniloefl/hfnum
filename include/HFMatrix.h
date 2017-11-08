#ifndef HF_H
#define HF_H

#include "Grid.h"
#include "Orbital.h"

#include <Eigen/Sparse>

// Objective: solve deriv(deriv(orbitals)) + g*orbitals = s using Newton-Raphson
// The orbitals are discretised in a Grid
// The equations to be solved are:
//   * for each orbital Grid point n: y_n+1 (1 + dx^2/12 g_n+1) - 2 y_n (1 - 5 dx^2/12 g_n) + y_n-1 (1 + dx^2/12 g_n-1) = dx^2 (s_n+1 + 10 s_n + s_n-1)
//   * sum orbital^2 r^2 dr = 1 -> guarantees normalisation of the orbitals
//   * |orbital energies| are minimised -> sum orbital Energy^2 = 0 -> should be the ground state to be minimised, really
// The set of equations can be summarised as:
//   * F_i(z_j) = 0
// This system is non-linear due to the energies and orbital normalisation requirement.
//
// The procedure is to get the derivatives of F in J and solve a linear system iteratively
//   * (++) Define J(i, j) = dF_i/dz_j
//   * F(z + dz) = F(z) + J . dz + O(dz^2)
//   * If we want F(x+dz) = 0:
//     * J.dz = -F(z)
//   * Define F0 = -F(z)
//   * Solve:
//     * J.dz = F0
//   * Update:
//     * z = z + gamma*dz
//   * Repeat (++)
class HF {
  public:
    HF(const Grid &g, double Z);
    virtual ~HF();

    void solve(int NiterSCF, int Niter, double F0stop);
    void solveForFixedPotentials(int Niter, double F0stop);
    void step();
    void prepareKinetic();
    void addOrbital(int L, int initial_n = 1, int initial_l = 0, int initial_m = 0);

    std::vector<double> getOrbital(int no, int mo, int lo);

    std::vector<double> getNucleusPotential();
    std::vector<double> getDirectPotential();

    void gammaSCF(double g);

  private:
    void calculateVd(double gamma);

    const Grid &_g;
    double _Z;
    std::vector<Orbital> _o;

    Eigen::SparseMatrix<double> _J;      // coefficient matrix
    Eigen::SparseMatrix<double> _F0;     // independent column matrix

    std::vector<double> _pot;            // effective potential
    std::vector<double> _potIndep;       // non-multiplicative part of the potential
    std::map<std::pair<int, int>, std::vector<double> > _vd; // vd

    double _nF0;                         // sum of F0 items -> to minimize
    Eigen::SparseMatrix<double> _dz;     // solution in a step
    
    double _gamma_scf;
};

#endif

