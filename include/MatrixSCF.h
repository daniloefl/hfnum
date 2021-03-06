/*
 * \class MatrixSCF
 *
 * \ingroup hfnum
 *
 * \brief Implements the self-consistent field calculation base class for methods using matricial bases.
 */

#ifndef MATRIXSCF_H
#define MATRIXSCF_H

#include <vector>
#include <map>

#include "utils.h"
#include "Basis.h"

class OrbitalQuantumNumbers {
  public:
  int n;
  int l;
  int m;
  int s;
  ldouble E;
  OrbitalQuantumNumbers(int in = 1, int il = 0, int im = 0, int is = 1, ldouble iE = -0.5);
  OrbitalQuantumNumbers(const OrbitalQuantumNumbers &o);
  OrbitalQuantumNumbers &operator =(const OrbitalQuantumNumbers &o);
  ~OrbitalQuantumNumbers();
};

class MatrixSCF {
  public:

    /// \brief Constructor for an atom.
    MatrixSCF();

    /// \brief Destructor.
    virtual ~MatrixSCF();

    /// \brief Solve Roothan-Hartree-Fock equation
    virtual void solveRoothan() = 0;

    /// \brief Recalculate SCF and call solveRoothan iteratively
    virtual void solve() = 0;

    /// \brief Get Z.
    /// \return Z
    ldouble Z();

    /// \brief Set Z.
    /// \param Z Z value.
    virtual void setZ(ldouble Z);

    /// \brief Add an orbital in internal _o list.
    /// \param n Quantum number n
    /// \param l Quantum number l
    /// \param m Quantum number m
    /// \param s Quantum number s
    void addOrbital(int n, int l, int m, int s);

    /// \brief Change gamma parameter for self-consistent step, used to slowly incorporate direct and exchange potential fields.
    /// \param g gamma parameter.
    void gammaSCF(ldouble g);

    /// \brief Get number of orbitals.
    /// \param s Spin up or down.
    /// \return Number of orbitals.
    int getNOrbitals(int s);

    /// \brief Get spectroscopic name of orbital.
    /// \param no Orbital index.
    /// \param s Spin up or down.
    /// \return Orbital name
    std::string getOrbitalName(int no, int s);

    /// \brief Get orbital quantum number n.
    /// \param no Orbital index.
    /// \param s Spin up or down.
    /// \return Orbital quantum number n
    int getOrbital_n(int no, int s);

    /// \brief Get orbital quantum number l.
    /// \param no Orbital index.
    /// \param s Spin up or down.
    /// \return Orbital quantum number l
    int getOrbital_l(int no, int s);

    /// \brief Get orbital quantum number m.
    /// \param no Orbital index.
    /// \param s Spin up or down.
    /// \return Orbital quantum number m
    int getOrbital_m(int no, int s);

    /// \brief Get orbital spin.
    /// \param no Orbital index.
    /// \param s Spin up or down.
    /// \return Orbital spin
    int getOrbital_s(int no, int s);

    /// \brief Get orbital energy.
    /// \param no Orbital index.
    /// \param s Spin up or down.
    /// \return Orbital energy
    ldouble getOrbital_E(int no, int s);

    /// \brief Set number of SCF iterations
    /// \param Nscf Number of iterations
    void Nscf(int Nscf);

  protected:

    /// \brief Set basis
    /// \param b Basis
    void setBasis(Basis *b);


    /// Atomic number
    ldouble _Z;

    /// Speed at which the new Vd and Vex are integrated into the next self-consistent step
    ldouble _gamma_scf;

    /// Number of SCF iterations
    int _Nscf_max;

    /// Orbital definition
    std::vector<OrbitalQuantumNumbers> _o_up;
    std::vector<OrbitalQuantumNumbers> _o_dw;
 
    /// Basis pointer
    Basis *_b;
};

#endif

