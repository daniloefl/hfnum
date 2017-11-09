#ifndef HF_H
#define HF_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

typedef std::map<std::pair<int, int>, std::vector<double> > Vd;
typedef std::map<std::pair<int, int>, std::vector<double> > Vex;

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

    void calculateFMatrix(std::vector<MatrixXd> &F, std::vector<double> &E);

    long double step();
    std::vector<double> solveOrbitalFixedEnergy(std::vector<double> &E, std::vector<int> &l, std::vector<MatrixXd> &Fm, std::vector<int> &icl);
    void addOrbital(int L, int s, int initial_n = 1, int initial_l = 0, int initial_m = 0);

    std::vector<double> getOrbital(int no, int mo, int lo);

    std::vector<double> getNucleusPotential();
    std::vector<double> getDirectPotential(int k);
    std::vector<double> getExchangePotential(int k);

    void gammaSCF(double g);

  private:
    void calculateVd(double gamma);
    void calculateVex(double gamma);

    void solveInward(std::vector<double> &E, std::vector<int> &l, std::vector<VectorXd> &solution, std::vector<MatrixXd> &Fm);
    void solveOutward(std::vector<double> &E, std::vector<int> &l, std::vector<VectorXd> &solution, std::vector<MatrixXd> &Fm);
    void match(std::vector<VectorXd> &o, std::vector<int> &icl, std::vector<VectorXd> &inward, std::vector<VectorXd> &outward);

    const Grid &_g;
    double _Z;
    std::vector<Orbital> _o;

    std::vector<double> _pot;            // effective potential
    std::vector<double> _potIndep;       // non-multiplicative part of the potential
    std::map<int, Vd>   _vd;             // vd
    std::map<int, Vex>  _vex;            // vex

    std::vector<double> _dE;             // resulting delta energy
    
    double _gamma_scf;
};

#endif

