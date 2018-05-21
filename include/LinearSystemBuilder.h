/*
 * \class LinearSystemBuilder
 *
 * \ingroup hfnum
 *
 * \brief Creates sparse matrix and solves equation assuming only initial condition at zero and infinity.
 */

#ifndef LINEARSYSTEMBUILDER_H
#define LINEARSYSTEMBUILDER_H

#include <Eigen/Core>
#include <vector>

#include "Orbital.h"
#include "Grid.h"
#include "utils.h"
#include "OrbitalMapper.h"

class LinearSystemBuilder {
  public:

    /// \brief Constructor.
    /// \param g Grid.
    /// \param o Orbitals.
    /// \param i Crossing points.
    /// \param om Orbital to index mapper.
    LinearSystemBuilder(const Grid &g, std::vector<Orbital *> &o, std::vector<int> &i, OrbitalMapper &om);

    /// \brief Destructor
    virtual ~LinearSystemBuilder();

    /// \brief Build sparse matrices to solve system of equations.
    /// \param A Jacobian matrix.
    /// \param b0 Column vector at current orbital and energy configuration.
    /// \param pot Coulomb potential.
    /// \param vd Direct potential.
    /// \param vex Exchange potential.
    /// \param lambda Lagrange multipliers.
    /// \param Map indicating which lagrange multipliers refer to which orbital.
    void prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<ldouble> &pot, std::map<int, Vradial> &vd, std::map<std::pair<int, int>, Vradial> &vex, std::vector<ldouble> &lambda, std::map<int, int> &lambdaMap);

    /// \brief Build sparse matrices to solve system of equations.
    /// \param A Jacobian matrix.
    /// \param b0 Column vector at current orbital and energy configuration.
    /// \param pot Coulomb potential.
    /// \param vsum Extra multiplicative potential (in DFT case).
    void prepareMatrices(SMatrixXld &A, VectorXld &b0, std::vector<ldouble> &pot, std::vector<ldouble> &vsum_up, std::vector<ldouble> &vsum_dw);

    /// \brief Propagate results of doing one step in the direction of -b0*Jacobian.inverse() to orbitals and energy vectors.
    /// \param b New solution.
    /// \param dE Energy step to be returned by reference.
    /// \param gamma Speed to go in solution direction.
    void propagate(VectorXld &b, std::vector<ldouble> &dE, const ldouble gamma);

  private:
    /// Grid
    const Grid &_g;

    /// Orbitals
    std::vector<Orbital *> &_o;

    /// Crossing points for reference
    std::vector<int> &icl;

    /// Orbital to index mapper
    OrbitalMapper &_om;
};

#endif

