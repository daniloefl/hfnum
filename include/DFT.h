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

    /// \brief Save orbital and potentials in file.
    /// \param fout File name on which to save information.
    void save(const std::string fout);

    /// \brief Load orbital and potentials from file.
    /// \param fin File name from which to load information.
    void load(const std::string fin);

    /// \brief Main method to solve system. Calculate potentials and call "solveForFixedPotentials".
    /// \param NiterSCF Number of self-consistent iterations.
    /// \param Niter Number of iterations when looking for correct energy eigenvalue.
    /// \param F0step Stop looking for correct energies, when all eigenvalues change by lass than this amount.
    void solve(int NiterSCF, int Niter, ldouble F0stop);

    /// \brief Keep self-consistent potentials constant and solve Schroedinger eq., looking for the correct energy eigenvalue.
    /// \param Niter Number of iterations used to seek for energy eigenvalue.
    /// \param F0stop When energies vary by less than this value, stop looking for energy eigenvalue.
    /// \return Value of a constraint being minimised (depends on the method).
    ldouble solveForFixedPotentials(int Niter, ldouble F0stop);

    /// \brief Use Gordon's method, which tries a set of orthogonal initial conditions to find the energy.
    /// \param gamma Factor used to regulate speed on which we go in the direction of the minimum when looking for energy eigenvalues.
    /// \return Minimisation function value at the end of the step.
    ldouble stepGordon(ldouble gamma);

    /// \brief Use renormalised wave function method, which looks for solution in ratio of Numerov parameters to avoid overflow.
    /// \param gamma Factor used to regulate speed on which we go in the direction of the minimum when looking for energy eigenvalues.
    /// \return Minimisation function value at the end of the step.
    ldouble stepRenormalised(ldouble gamma);

    ldouble solveOrbitalFixedEnergy(std::vector<ldouble> &E, std::vector<int> &l, std::vector<MatrixXld> &Fm, std::vector<MatrixXld> &Km, std::vector<VectorXld> &matched);

    /// \brief Build NxN matrix to solve all equations of the Numerov method for each point simultaneously. Includes an extra equation to control the orbital normalisations, which is non-linear.
    /// \param gamma Factor used to regulate speed on which we go in the direction of the minimum when looking for energy eigenvalues.
    /// \return Minimisation function value at the end of the step.
    ldouble stepSparse(ldouble gamma);

    /// \brief Calculate total energy
    /// \return Ground energy
    ldouble getE0();

    /// \brief Calculate F matrix, which represents the Hamiltonian using the Numerov method. K is the inverse of F. This is used for the Gordon and renormalised methods, since these matrices are calculated per Grid point. The sparse method uses a large matrix solving all points simultaneously.
    /// \param F To be returned by reference. Matrix F for each Grid point.
    /// \param K To be returned by reference. Inverse of F.
    /// \param C To be returned by reference. Independent term.
    /// \param E Values of energy in each orbital.
    void calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<MatrixXld> &C, std::vector<ldouble> &E);

    /// \brief Add an orbital in internal _o list.
    /// \param o Pointer to orbital.
    void addOrbital(Orbital *o);

    /// \brief Get electron density for spin up orbitals.
    /// \return Vector of electron density values for each Grid point.
    std::vector<ldouble> getDensityUp();

    /// \brief Get electron density for spin up orbitals.
    /// \return Vector of electron density values for each Grid point.
    boost::python::list getDensityUpPython();

    /// \brief Get Hartree potential
    /// \return Vector of hartree potential values for each Grid point.
    boost::python::list getHartreePython();

    /// \brief Get Exchange potential for spin up orbitals
    /// \return Vector of exchange potential up values for each Grid point.
    boost::python::list getExchangeUpPython();

    /// \brief Get Exchange potential for spin down orbitals
    /// \return Vector of exchange potential down values for each Grid point.
    boost::python::list getExchangeDownPython();

    /// \brief Get electron density for spin down orbitals.
    /// \return Vector of electron density values for each Grid point.
    std::vector<ldouble> getDensityDown();

    /// \brief Get electron density for spin down orbitals.
    /// \return Vector of electron density values for each Grid point.
    boost::python::list getDensityDownPython();

  private:
    /// \brief Calculate direct SCF potentials.
    /// \param gamma Parameter used to take a linear combination of previous potential and new one.
    void calculateV(ldouble gamma);

    /// \brief Calculate electron density
    /// \param gamma Parameter used to take a linear combination of previous potential and new one.
    void calculateN(ldouble gamma);

    /// Electron densities
    std::vector<ldouble> _nsum_up;
    std::vector<ldouble> _n_up;

    std::vector<ldouble> _nsum_dw;
    std::vector<ldouble> _n_dw;

    /// central Hartree potential
    std::vector<ldouble> _u;

    /// LDA exchange potential
    std::vector<ldouble> _vex_lda_up;
    std::vector<ldouble> _vex_lda_dw;

    /// total potential
    std::vector<ldouble> _vsum_up;
    std::vector<ldouble> _vsum_dw;
};

#endif

