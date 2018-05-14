/*
 * \class HF
 *
 * \ingroup hfnum
 *
 * \brief Implements the Hartree-Fock calculation entry point, which holds results and calls specific calculation methods.
 */

#ifndef HF_H
#define HF_H

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

class HF : public SCF {
  public:

    /// \brief Constructor for an atom.
    HF();

    /// \brief Constructor for an atom.
    /// \param fname Input result form previous calculation for plotting
    HF(const std::string fname);

    /// \brief Destructor.
    virtual ~HF();

    /// \brief Get direct potential
    /// \param k Identifier of the orbital on which this potential is to be applied.
    /// \return Direct potential.
    std::vector<ldouble> getDirectPotential(int k);

    /// \brief Get exchange potential
    /// \param k Identifier of the orbital equation on which this potential is to be applied.
    /// \param k2 Identifier of the orbital on which this potential is to be applied.
    /// \return Exchange potential.
    std::vector<ldouble> getExchangePotential(int k, int k2);

    /// \brief Get direct potential. Python interface.
    /// \param k Identifier of the orbital on which this potential is to be applied.
    /// \return Direct potential.
    boost::python::list getDirectPotentialPython(int k);

    /// \brief Get exchange potential. Python interface.
    /// \param k Identifier of the orbital equation on which this potential is to be applied.
    /// \param k2 Identifier of the orbital on which this potential is to be applied.
    /// \return Exchange potential.
    boost::python::list getExchangePotentialPython(int k, int k2);

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

    /// \brief Calculate total energy
    /// \return Ground energy
    ldouble getE0();

    /// \brief Add an orbital in internal _o list.
    /// \param o Pointer to orbital.
    void addOrbital(Orbital *o);

  protected:

    /// \brief Calculate F matrix, which represents the Hamiltonian using the Numerov method. K is the inverse of F. This is used for the Gordon and renormalised methods, since these matrices are calculated per Grid point. The sparse method uses a large matrix solving all points simultaneously.
    /// \param F To be returned by reference. Matrix F for each Grid point.
    /// \param K To be returned by reference. Inverse of F.
    /// \param E Values of energy in each orbital.
    void calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E);

  private:

    // use average coefficients for non-filled groups
    bool _averageCoefficients;

    /// \brief Calculate aux. variable Y.
    void calculateY();

    /// \brief Calculate direct SCF potentials.
    /// \param gamma Parameter used to take a linear combination of previous potential and new one.
    void calculateVd(ldouble gamma);

    /// \brief Calculate exchange SCF potentials.
    /// \param gamma Parameter used to take a linear combination of previous potential and new one.
    void calculateVex(ldouble gamma);

    /// Auxiliary variables Y and Z
    std::map<int, Vradial> _Y;
    std::map<int, Vradial> _Zt;

    /// temporary variable for the new Vd
    std::map<int, Vradial>   _vdsum;

    /// temporary variable for the new Vex
    std::map<std::pair<int, int>, Vradial>  _vexsum;

};

#endif

