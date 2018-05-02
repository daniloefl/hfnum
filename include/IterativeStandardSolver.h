/*
 * \class IterativeStandardSolver
 *
 * \ingroup hfnum
 *
 * \brief Solves the 2nd degree differential equation using Numerov method assuming a non-homogeneous term. Resolves it many times with the same energy and direct and exchange terms to include all potential terms consistently.
 */

#ifndef ITERATIVESTANDARDSOLVER_H
#define ITERATIVESTANDARDSOLVER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include <vector>
#include "OrbitalMapper.h"

class IterativeStandardSolver {
  public:

    /// \brief Constructor
    /// \param g Grid.
    /// \param o List of orbitals.
    /// \param i Grid positions where to make the matching.
    /// \param om Class to provide orbital number to matrix index mapping.
    /// \param Z Atomic number used for initial condition at infinity.
    IterativeStandardSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om, ldouble Z = 1.0);

    /// \brief Destructor.
    virtual ~IterativeStandardSolver();

    /// \brief Solve equation for a specific energy.
    /// \param E Trial energy.
    /// \param pot Nucleus potential.
    /// \param vd Direct potential for each orbital.
    /// \param vex Exchange potential for each orbital.
    /// \param matched To be returned by reference.
    /// \param c Convergence acceleration parameter.
    /// \return Minimisation function based on matching at classical crossing for the trial energy.
    VectorXld solve(std::vector<ldouble> &E, Vradial &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::map<int, Vradial> &matched, ldouble c = 1.0);

    /// \brief Solve equation for a specific energy.
    /// \param E Trial energy.
    /// \param pot Nucleus potential.
    /// \param vup Potential only on up electrons.
    /// \param vdw Potential only on down electrons.
    /// \param matched To be returned by reference.
    /// \return Minimisation function based on matching at classical crossing for the trial energy.
    VectorXld solve(std::vector<ldouble> &E, Vradial &pot, Vradial &vup, Vradial &vdw, std::map<int, Vradial> &matched);

    /// \brief Solve assuming initial conditions at the 2 last grid points.
    /// \param E Trial energy.
    /// \param matched Previous orbitals to be used for non-homogenous term.
    /// \param idx Index of orbital to solve.
    /// \param solution To be returned by reference.
    void solveInward(std::vector<ldouble> &E, std::map<int, Vradial> &matched, int idx, Vradial &solution);

    /// \brief Solve assuming initial conditions at the 2 first grid points.
    /// \param E Trial energy.
    /// \param matched To be returned by reference. Orbitals found.
    void solveOutward(std::vector<ldouble> &E, std::map<int, Vradial> &matched, int idx, Vradial &solution);

    /// \brief Force continuity by taking ratio of inward and outward solutions at the matching point and scaling the solutions appropriately.
    /// \param k Index of the orbital.
    /// \param o Matched orbitals.
    /// \param inward Inward solution.
    /// \param outward Outward solution.
    /// \param c Convergence acceleration parameter
    void match(int k, Vradial &o, Vradial &inward, Vradial &outward, ldouble c = 1.0);

    /// \brief Set Z value.
    /// \param Z New atomic number.
    void setZ(ldouble Z);

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
    std::map<int, Vradial> s;

    std::map<int, Vradial> inward;
    std::map<int, Vradial> outward;

    ldouble _Z;
};

#endif

