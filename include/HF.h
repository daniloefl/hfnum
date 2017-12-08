#ifndef HF_H
#define HF_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>
#include <map>

#include "utils.h"
#include "LinearSystemBuilder.h"
#include "IterativeRenormalisedSolver.h"
#include "IterativeGordonSolver.h"
#include "OrbitalMapper.h"


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

    // main method to solve system
    // calculate potentials and call "solveForFixedPotentials"
    void solve(int NiterSCF, int Niter, ldouble F0stop);

    // fix self-consistent potentials and call step or stepSparse to solve Schroedinger eq.
    ldouble solveForFixedPotentials(int Niter, ldouble F0stop);

    // methods to solve equation for a fixed potential
    // This uses the Hydrogen initial conditions for the solution
    // Does not seem to work for Li or Be
    // Sparse method in stepSparse is more general
    ldouble stepGordon(ldouble gamma);

    ldouble stepRenormalised(ldouble gamma);

    ldouble solveOrbitalFixedEnergy(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched);

    // calculate F matrix and its inverse K
    // this contains the coefficients
    void calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E);

    // build NxN matrix to solve all equations of the Numerov method for each point simultaneously
    // includes an extra equation to control the orbital normalisations, which is non-linear
    ldouble stepSparse(ldouble gamma);

    // add orbital
    void addOrbital(int s, int initial_n = 1, int initial_l = 0, int initial_m = 0);

    // change SCF speed
    void gammaSCF(ldouble g);

    // change whether to use sparse method
    void method(int m);

    // getters
    std::vector<ldouble> getOrbital(int no, int mo, int lo);

    std::vector<ldouble> getNucleusPotential();
    std::vector<ldouble> getDirectPotential(int k);
    std::vector<ldouble> getExchangePotential(int k, int k2);


  private:
    // calculate SCF potentials
    void calculateVd(ldouble gamma);
    void calculateVex(ldouble gamma);

    // solve equation with fixed potentials and energy assuming initial condition at infinity
    void solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R);

    // solve equation with fixed potentials and energy assuming initial condition at r=0
    void solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &solution, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R);

    // match solutions of solveInward and solveOutward at the position given by icl
    void match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward);

    // numerical Grid
    const Grid &_g;

    // atomic number
    ldouble _Z;

    // vector of orbitals
    std::vector<Orbital> _o;

    // effective potential
    std::vector<ldouble> _pot;

    // Direct potential
    std::map<int, Vd>   _vd;

    // Exchange potential
    std::map<std::pair<int, int>, Vex>  _vex;

    // variation in energy for the next step of step() or stepSparse()
    std::vector<ldouble> _dE;
    
    // speed at which the new Vd and Vex are integrated into the next self-consistent step
    ldouble _gamma_scf;

    // temporary variable for the new Vd
    std::map<int, Vd>   _vdsum;
    // temporary variable for the new Vex
    std::map<std::pair<int, int>, Vex>  _vexsum;

    // index in the numerical Grid where the classical crossing of energy-potential happens
    // used in the non-sparse method for the dE evaluation
    std::vector<int> icl;

    // record maximum energy if needed
    std::vector<ldouble> _Emax;

    // record minimum energy if needed
    std::vector<ldouble> _Emin;

    // record number of zero crossings of the main orbital function if needed
    std::vector<int> _nodes;

    // whether the method for solving the equation should be sparse
    int _method;

    // for the matrix Numerov method using sparse matrices
    // A is the Jacobian of the non-linear system
    SMatrixXld _A;
    // b is the column vector of solutions
    VectorXld _b;
    // b0 is the column vector of last solutions for the iteration
    VectorXld _b0;

    LinearSystemBuilder _lsb;
    IterativeRenormalisedSolver _irs;
    IterativeGordonSolver _igs;
    OrbitalMapper _om;
};

#endif

