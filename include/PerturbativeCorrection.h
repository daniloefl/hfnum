/*
 * \class PerturbativeCorrection
 *
 * \ingroup hfnum
 *
 * \brief Takes the result of a previous atomic calculation and applies a perturbative correction to it.
 */

#ifndef PERTURBATIVECORRECTION_H
#define PERTURBATIVECORRECTION_H

#include "utils.h"
#include "Orbital.h"
#include <vector>
#include <string>
#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include "Grid.h"
#include <Python.h>
#include <Eigen/Core>
#include <fstream>

class PerturbativeCorrection {

  public:

    /// \brief Default constructor
    PerturbativeCorrection();

    /// \brief Destructor
    virtual ~PerturbativeCorrection();

    /// \brief Function to be implemented to load information from a previous SCF or perturbative calculation
    /// \param fname Saved file name
    virtual void load(const std::string &fname) = 0;

    /// \brief Function to be implemented to calculate perturbative correction
    virtual void correct() = 0;

    /// \brief Get list of R values in the Grid
    /// \return List of R values in the Grid
    boost::python::list getR();

    /// \brief Get corrected eigenenergies
    /// \return List of corrected energies
    std::vector<ldouble> getCorrectedE();

    /// \brief Get corrected eigenenergies
    /// \return List of corrected energies
    boost::python::list getCorrectedEPython();

    /// \brief Get matrix showing mixing of original eigen-states after correction
    /// \return Coefficients matrix
    MatrixXld getCoefficients();

    /// \brief Get matrix showing mixing of original eigen-states after correction
    /// \return Coefficients matrix
    boost::python::list getCoefficientsPython();

    /// \brief Get value of orbital component for orbital no, assuming a central potential.
    /// \param no Orbital identification.
    /// \return Vector of orbital values for each Grid point, in that spherical harmonic component.
    std::vector<ldouble> getOrbitalCentral(int no);

    /// \brief Get value of orbital component for orbital no, assuming a central potential.
    /// \param no Orbital identification.
    /// \return Vector of orbital values for each Grid point, in that spherical harmonic component.
    boost::python::list getOrbitalCentralPython(int no);

    /// \brief Get number of orbitals.
    /// \return Number of orbitals.
    int getNOrbitals();

    /// \brief Get spectroscopic name of orbital.
    /// \param no Orbital index.
    /// \return Orbital name
    std::string getOrbitalName(int no);

    /// \brief Get orbital quantum number n.
    /// \param no Orbital index.
    /// \return Orbital quantum number n
    int getOrbital_n(int no);

    /// \brief Get orbital quantum number l.
    /// \param no Orbital index.
    /// \return Orbital quantum number l
    int getOrbital_l(int no);

    /// \brief Get orbital quantum number m.
    /// \param no Orbital index.
    /// \return Orbital quantum number m
    int getOrbital_m(int no);

    /// \brief Get orbital spin.
    /// \param no Orbital index.
    /// \return Orbital spin
    int getOrbital_s(int no);

    /// \brief Get orbital energy.
    /// \param no Orbital index.
    /// \return Orbital energy
    ldouble getOrbital_E(int no);

    /// \brief Get atomic number
    /// \return Atomic number
    ldouble Z();

  protected:
    /// List of owned Orbital pointers
    std::vector<Orbital *> _o;
    
    /// Numerical Grid
    Grid *_g;

    /// Atomic number
    ldouble _Z;

    /// Energies
    std::vector<ldouble> _E;

    /// Corrected energies
    std::vector<ldouble> _Ec;

    /// Coefficients matrix
    MatrixXld _c;
    
};

#endif

