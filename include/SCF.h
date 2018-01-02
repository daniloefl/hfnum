/*
 * \class SCF
 *
 * \ingroup hfnum
 *
 * \brief Implements the self-consistent field calculation base class, which holds results and calls specific calculation methods.
 */

#ifndef SCF_H
#define SCF_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>
#include <map>

#include "utils.h"
#include "LinearSystemBuilder.h"
#include "IterativeRenormalisedSolver.h"
#include "IterativeGordonSolver.h"
#include "OrbitalMapper.h"

#include <Python.h>
using namespace boost;

class SCF {
  public:

    /// \brief Constructor for an atom.
    SCF();

    /// \brief Constructor for an atom.
    /// \param fname Input result form previous calculation for plotting
    SCF(const std::string fname);

    /// \brief Constructor for an atom.
    /// \param g Grid object.
    /// \param Z Atomic number.
    SCF(Grid &g, ldouble Z);

    /// \brief Constructor for an atom.
    /// \param o Grid object for a Python interface.
    /// \param Z Atomic number.
    SCF(python::object o, ldouble Z);

    /// \brief Destructor.
    virtual ~SCF();

    /// \brief Reset Grid configuration.
    /// \param isLog Whether the Grid is logarithmic.
    /// \param dx Grid step.
    /// \param N Number of Grid points.
    /// \param rmin Minimum Grid r value.
    void resetGrid(bool isLog, ldouble dx, int N, ldouble rmin);

    /// \brief Get Z.
    /// \return Z
    ldouble Z();

    /// \brief Set Z.
    /// \param Z Z value.
    void setZ(ldouble Z);

    /// \brief Get Grid.
    /// \return Grid object.
    Grid &getGrid();

    /// \brief Get list of R values from the Grid.
    /// \return List of R values
    boost::python::list getR() const;

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


    /// \brief Calculate F matrix, which represents the Hamiltonian using the Numerov method. K is the inverse of F. This is used for the Gordon and renormalised methods, since these matrices are calculated per Grid point. The sparse method uses a large matrix solving all points simultaneously.
    /// \param F To be returned by reference. Matrix F for each Grid point.
    /// \param K To be returned by reference. Inverse of F.
    /// \param E Values of energy in each orbital.
    virtual void calculateFMatrix(std::vector<MatrixXld> &F, std::vector<MatrixXld> &K, std::vector<ldouble> &E) = 0;

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

    /// \brief Add an orbital in internal _o list.
    /// \param o Pointer to orbital.
    virtual void addOrbital(Orbital *o);

    /// \brief Change gamma parameter for self-consistent step, used to slowly incorporate direct and exchange potential fields.
    /// \param g gamma parameter.
    void gammaSCF(ldouble g);

    /// \brief Select method to be used.
    /// \param m Can be 0 for the sparse method, 1 for the Gordon method, 2 for the renormalised method.
    void method(int m);

    /// \brief Get value of orbital component for orbital no, in spherical harmonic given by lo and mo.
    /// \param no Orbital identification.
    /// \param lo Spherical harmonic parameter l.
    /// \param mo Spherical harmonic parameter m.
    /// \return Vector of orbital values for each Grid point, in that spherical harmonic component.
    std::vector<ldouble> getOrbital(int no, int mo, int lo);

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

    /// \brief Get Coulomb attraction potential -Z/r
    /// \return Coulomb potential
    std::vector<ldouble> getNucleusPotential();

    /// \brief Get Coulomb attraction potential -Z/r. Python interface.
    /// \return Coulomb potential
    boost::python::list getNucleusPotentialPython();

    /// \brief Add an orbital from a Python object.
    /// \param o Orbital object from Python interface.
    void addOrbitalPython(boost::python::object o);

    /// \brief Force the direct and exchange potential calculation to assume only a central potential.
    /// \param central Whether to consider a central potential
    void centralPotential(bool central);

    /// Direct potential
    std::map<int, Vd>   _vd;

    /// Exchange potential
    std::map<std::pair<int, int>, Vex>  _vex;

    /// temporary variable for the new Vd
    std::map<int, Vd>   _vdsum;

    /// temporary variable for the new Vex
    std::map<std::pair<int, int>, Vex>  _vexsum;


  protected:
    /// Numerical Grid
    Grid *_g;

    /// Atomic number
    ldouble _Z;

    /// Vector of orbitals
    std::vector<Orbital *> _o;

    /// Coulomb potential
    std::vector<ldouble> _pot;

    /// variation in energy for the next step
    std::vector<ldouble> _dE;
    
    /// Speed at which the new Vd and Vex are integrated into the next self-consistent step
    ldouble _gamma_scf;

    /// index in the numerical Grid where the classical crossing of energy-potential happens
    std::vector<int> icl;

    /// record maximum energy if needed
    std::vector<ldouble> _Emax;

    /// record minimum energy if needed
    std::vector<ldouble> _Emin;

    /// record number of zero crossings of the main orbital function if needed
    std::vector<int> _nodes;

    /// Method to be used for solving equations.
    int _method;

    /// for the matrix Numerov method using sparse matrices
    /// A is the Jacobian of the non-linear system
    SMatrixXld _A;
    /// b is the column vector of solutions
    VectorXld _b;
    /// b0 is the column vector of last solutions for the iteration
    VectorXld _b0;

    /// Solver using sparse matrices
    LinearSystemBuilder _lsb;

    /// Solver for the renormalised method
    IterativeRenormalisedSolver _irs;

    /// Solver for Gordon's method
    IterativeGordonSolver _igs;

    /// Class that maps orbitals to indices
    OrbitalMapper _om;

    /// Central potential?
    bool _central;

    /// List of owned Orbital pointers
    std::vector<Orbital *> _owned_orb;

    /// Whether we own the Grid
    bool _own_grid;
};

#endif

