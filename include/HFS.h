/*
 * \class HFS
 *
 * \ingroup hfnum
 *
 * \brief Implements the Hartree-Fock-Slater calculation entry point, which holds results and calls specific calculation methods.
 */

#ifndef HFS_H
#define HFS_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>
#include <map>

#include "SCF.h"
#include "utils.h"
#include "LinearSystemBuilder.h"
#include "OrbitalMapper.h"

#include <Python.h>
using namespace boost;

class HFS : public SCF {
  public:

    /// \brief Constructor for an atom.
    HFS(ldouble Z = 1);

    /// \brief Constructor for an atom.
    /// \param fname Input result form previous calculation for plotting
    HFS(const std::string fname);

    /// \brief Destructor.
    virtual ~HFS();

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

  private:

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

