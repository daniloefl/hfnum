/*
 * \class DFT
 *
 * \ingroup hfnum
 *
 * \brief Implements the DFT calculation entry point, which holds results and calls specific calculation methods.
 */

#ifndef DFT_H
#define DFT_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>
#include <map>

#include "SCF.h"
#include "utils.h"
#include "LinearSystemBuilder.h"
#include "IterativeRenormalisedSolver.h"
#include "IterativeGordonSolver.h"
#include "OrbitalMapper.h"

#include <Python.h>
using namespace boost;

class DFT : public SCF {
  public:

    /// \brief Constructor for an atom.
    DFT();

    /// \brief Constructor for an atom.
    /// \param fname Input result form previous calculation for plotting
    DFT(const std::string fname);

    /// \brief Destructor.
    virtual ~DFT();

    /// \brief Main method to solve system. Calculate potentials and call "solveForFixedPotentials".
    /// \param NiterSCF Number of self-consistent iterations.
    /// \param Niter Number of iterations when looking for correct energy eigenvalue.
    /// \param F0step Stop looking for correct energies, when all eigenvalues change by lass than this amount.
    void solve(int NiterSCF, int Niter, ldouble F0stop);

    /// \brief Calculate total energy
    /// \return Ground energy
    ldouble getE0();

    /// \brief Calculate F matrix, which represents the Hamiltonian using the Numerov method. K is the inverse of F. This is used for the Gordon and renormalised methods, since these matrices are calculated per Grid point. The sparse method uses a large matrix solving all points simultaneously.
    /// \param F To be returned by reference. Matrix F for each Grid point.
    /// \param K To be returned by reference. Inverse of F.
    /// \param E Values of energy in each orbital.
    void calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E);

    /// \brief Add an orbital in internal _o list.
    /// \param o Pointer to orbital.
    void addOrbital(Orbital *o);

  private:
    /// \brief Calculate direct SCF potentials.
    /// \param gamma Parameter used to take a linear combination of previous potential and new one.
    void calculateVd(ldouble gamma);

    /// \brief Calculate electron density
    /// \param gamma Parameter used to take a linear combination of previous potential and new one.
    void calculateN(ldouble gamma);

    /// Electron densities
    Vd _nsum_up;
    Vd _n_up;

    Vd _nsum_dw;
    Vd _n_dw;

    // potential
    std::vector<ldouble> _u;
};

#endif

