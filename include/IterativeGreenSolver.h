/*
 * \class IterativeGreenSolver
 *
 * \ingroup hfnum
 *
 * \brief Solves the 2nd degree differential equation using Numerov method with the Green's function approach. Resolves it many times with the same energy and direct and exchange terms to include all potential terms consistently.
 */

#ifndef ITERATIVEGREENSOLVER_H
#define ITERATIVEGREENSOLVER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include <vector>
#include "OrbitalMapper.h"

class IterativeGreenSolver {
  public:

    /// \brief Constructor
    /// \param g Grid.
    /// \param o List of orbitals.
    /// \param i Grid positions where to make the matching.
    /// \param om Class to provide orbital number to matrix index mapping.
    /// \param Z Atomic number used for initial condition at infinity.
    IterativeGreenSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om, ldouble Z = 1.0);

    /// \brief Destructor.
    virtual ~IterativeGreenSolver();

    /// \brief Solve equation for a specific energy.
    /// \param E Trial energy.
    /// \param pot Nucleus potential.
    /// \param vd Direct potential for each orbital.
    /// \param vex Exchange potential for each orbital.
    /// \param lambda List of Lagrange multipliers used to maintain orthogonality between same l orbitals.
    /// \param lambdaMap Map establishing which lambda indices relate to a pair of orbital indices. The key is 100*k1 + k2.
    /// \param matched To be returned by reference.
    /// \return Minimisation function based on matching at classical crossing for the trial energy.
    VectorXld solve(std::vector<ldouble> &E, Vradial &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched);

    /// \brief Solve equation for a specific energy.
    /// \param E Trial energy.
    /// \param pot Nucleus potential.
    /// \param vup Potential only on up electrons.
    /// \param vdw Potential only on down electrons.
    /// \param lambda List of Lagrange multipliers used to maintain orthogonality between same l orbitals.
    /// \param lambdaMap Map establishing which lambda indices relate to a pair of orbital indices. The key is 100*k1 + k2.
    /// \param matched To be returned by reference.
    /// \return Minimisation function based on matching at classical crossing for the trial energy.
    VectorXld solve(std::vector<ldouble> &E, Vradial &pot, Vradial &vup, Vradial &vdw, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap, std::map<int, Vradial> &matched);

    /// \brief Solve assuming initial conditions at the 2 last grid points.
    /// \param E Trial energy.
    /// \param idx Index of orbital to solve.
    /// \param solution To be returned by reference.
    void solveInward(std::vector<ldouble> &E, int idx, Vradial &solution);

    /// \brief Solve assuming initial conditions at the 2 first grid points.
    /// \param E Trial energy.
    /// \param idx Index of orbital to solve.
    /// \param solution To be returned by reference.
    void solveOutward(std::vector<ldouble> &E, int idx, Vradial &solution);

    /// \brief Force continuity by taking ratio of inward and outward solutions at the matching point.
    /// \param k Index of the orbital.
    /// \param o Matched orbitals.
    /// \param inward Inward solution.
    /// \param outward Outward solution.
    void match(int k, Vradial &o, Vradial &inward, Vradial &outward);

    /// \brief Set normalisation to 1.
    /// \param k Index of the orbital.
    /// \param o Matched orbitals.
    void normalise(int k, Vradial &o);

    /// \brief Set Z value.
    /// \param Z New atomic number.
    void setZ(ldouble Z);

    std::vector<ldouble> _i0;
    std::vector<ldouble> _i1;

  private:

    /// Grid.
    const Grid &_g;

    /// Vector of orbitals
    std::vector<Orbital *> &_o;

    /// Crossing points
    std::vector<int> &icl;

    /// Orbital to index mapper
    OrbitalMapper &_om;

    /// auxiliary variables
    std::map<int, Vradial> f;

    std::map<int, Vradial> inward;
    std::map<int, Vradial> outward;
    std::map<int, Vradial> homogeneousSolution;
    std::map<int, Vradial> S;

    ldouble _Z;
};

#endif

