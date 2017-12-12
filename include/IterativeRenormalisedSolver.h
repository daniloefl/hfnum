/*
 * \class IterativeRenormalisedSolver
 *
 * \ingroup hfnum
 *
 * \brief Implements renormalised method, solving for the ratio of orbitals instead of orbitals themselves. Avoids overflow.
 */

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

    /// \brief Constructor
    /// \param g Grid.
    /// \param o List of orbitals.
    /// \param i Grid positions where to make the matching.
    /// \param om Class to provide orbital number to matrix index mapping.
    IterativeRenormalisedSolver(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om);

    /// \brief Destructor.
    virtual ~IterativeRenormalisedSolver();

    /// \brief Solve equation for a specific energy.
    /// \param E Trial energy.
    /// \param l Spherical harmonic l parameters.
    /// \param Fmn F matrix.
    /// \param Kmn K matrix.
    /// \param matched To be returned by reference. Orbitals found.
    /// \return Minimisation function based on matching at classical crossing for the trial energy.
    ldouble solve(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fmn, std::vector<MatrixXld> &Kmn, std::vector<VectorXld> &matched);

    /// \brief Solve assuming initial conditions at the 2 last grid points.
    /// \param E Trial energy.
    /// \param l Spherical harmonic l parameters.
    /// \param Fm F matrix.
    /// \param Km K matrix.
    /// \param R Orbital ratio matrix for different initial condition assumptions.
    void solveInward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R);

    /// \brief Solve assuming initial conditions at the 2 first grid points.
    /// \param E Trial energy.
    /// \param l Spherical harmonic l parameters.
    /// \param Fm F matrix.
    /// \param Km K matrix.
    /// \param R Orbital ratio matrix for different initial condition assumptions.
    void solveOutward(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<MatrixXld> &R);

    /// \brief Force continuity by taking ratio of inward and outward solutions at the matching point and scaling the solutions appropriately.
    /// \param o Matched orbitals.
    /// \param inward Inward solution.
    /// \param outward Outward solution.
    void match(std::vector<VectorXld> &o, std::vector<VectorXld> &inward, std::vector<VectorXld> &outward);

  private:

    /// Grid.
    const Grid &_g;

    /// Vector of orbitals
    std::vector<Orbital *> &_o;

    /// Crossing points
    std::vector<int> &icl;

    /// Orbital to index mapper
    OrbitalMapper &_om;

    /// Which crossing point to use (can only use one to build R matrix)
    int kl;
};

#endif

