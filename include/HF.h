#ifndef HF_H
#define HF_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

typedef long double ldouble;
typedef Matrix<ldouble, Dynamic, Dynamic> MatrixXld;
typedef Matrix<ldouble, Dynamic, 1> VectorXld;

typedef std::map<std::pair<int, int>, std::vector<ldouble> > Vd;
typedef std::map<std::pair<int, int>, std::vector<ldouble> > Vex;


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
    HF(const Grid &g, ldouble Z);
    virtual ~HF();

    void solve(int NiterSCF, int Niter, ldouble F0stop);
    void solveForFixedPotentials(int Niter, ldouble F0stop);

    void calculateFMatrix(std::vector<MatrixXld> &F, std::vector<ldouble> &E);

    ldouble step();
    std::vector<ldouble> solveOrbitalFixedEnergy(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<int> &icl);
    void addOrbital(int L, int s, int initial_n = 1, int initial_l = 0, int initial_m = 0);

    std::vector<ldouble> getOrbital(int no, int mo, int lo);

    std::vector<ldouble> getNucleusPotential();
    std::vector<ldouble> getDirectPotential(int k);
    std::vector<ldouble> getExchangePotential(int k, int k2);

    void gammaSCF(ldouble g);

  private:
    void calculateVd(ldouble gamma);
    void calculateVex(ldouble gamma);

    void solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm);
    void solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<VectorXld> &solution, std::vector<MatrixXld> &Fm);
    void match(std::vector<VectorXld> &o, std::vector<int> &icl, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward);

    const Grid &_g;
    ldouble _Z;
    std::vector<Orbital> _o;

    std::vector<ldouble> _pot;            // effective potential
    std::vector<ldouble> _potIndep;       // non-multiplicative part of the potential
    std::map<int, Vd>   _vd;             // vd
    std::map<std::pair<int, int>, Vex>  _vex; // vex

    std::vector<ldouble> _dE;             // resulting delta energy
    
    ldouble _gamma_scf;

    ldouble _norm;
};

#endif

