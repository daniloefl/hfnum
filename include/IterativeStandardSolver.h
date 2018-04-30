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
    IterativeStandardSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om);

    /// \brief Destructor.
    virtual ~IterativeStandardSolver();

    /// \brief Solve equation for a specific energy.
    /// \param E Trial energy.
    /// \param l Spherical harmonic l parameters.
    /// \param vd Direct potential for each orbital.
    /// \param vex Exchange potential for each orbital.
    /// \param matched To be returned by reference.
    /// \return Minimisation function based on matching at classical crossing for the trial energy.
    ldouble solve(std::vector<ldouble> &E, std::vector<int> &l,  std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<Vradial> &matched);

    /// \brief Solve assuming initial conditions at the 2 last grid points.
    /// \param E Trial energy.
    /// \param l Spherical harmonic l parameters.
    /// \param vd Direct potential for each orbital.
    /// \param vex Exchange potential for each orbital.
    /// \param matched Previous orbitals to be used for non-homogenous term.
    /// \param idx Index of orbital to solve.
    /// \param solution To be returned by reference.
    void solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<Vradial> &matched, int idx, Vradial &solution);

    /// \brief Solve assuming initial conditions at the 2 first grid points.
    /// \param E Trial energy.
    /// \param l Spherical harmonic l parameters.
    /// \param vd Direct potential for each orbital.
    /// \param vex Exchange potential for each orbital.
    /// \param matched To be returned by reference. Orbitals found.
    void solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<Vradial> &matched, int idx, Vradial &solution);

    /// \brief Force continuity by taking ratio of inward and outward solutions at the matching point and scaling the solutions appropriately.
    /// \param k Index of the orbital.
    /// \param o Matched orbitals.
    /// \param inward Inward solution.
    /// \param outward Outward solution.
    void match(int k, Vradial &o, Vradial &inward, Vradial &outward);

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
    std::vector<Vradial> f;
    std::vector<Vradial> s;

};

#endif

