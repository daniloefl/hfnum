/*
 * \class Orbital
 *
 * \ingroup hfnum
 *
 * \brief Class that represents a single orbital in Hartree-Fock method.
 */

#ifndef ORBITAL_H
#define ORBITAL_H

#include <map>
#include <vector>

#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>

#include <Python.h>
using namespace boost;

class Grid;

typedef std::pair<int, int> lm;

#include "utils.h"

class Orbital {
  public:

    /// \brief Constructor.
    /// \param s Spin.
    /// \param initial_n Quantum number n. For initial conditions.
    /// \param initial_l Quantum number l. For initial conditions.
    /// \param initial_m Quantum number m. For initial conditions.
    Orbital(int s = 1, int initial_n = 1, int initial_l = 0, int initial_m = 0);

    /// \brief Constructor.
    /// \param o Object to copy.
    Orbital(const Orbital &o);

    /// \brief Destructor.
    virtual ~Orbital();

    /// \brief Assignment operator.
    /// \param o Orbital to copy.
    /// \return This.
    Orbital &operator =(const Orbital &o);

    /// \brief Add a spherical harmonic component.
    /// \param l Value of l.
    /// \param m Value of m.
    void addSphHarm(int l, int m);

    /// \brief Getter for list of spherical harmonic factors.
    /// \return List of (l, m) pairs
    const std::vector<lm> &getSphHarm() const;
     
    /// \brief Set number of Grid points.
    /// \param N number of Grid points.
    void N(int N);

    /// \brief Getter for number of Grid points.
    /// \return Number of Grid points.
    int N() const;

    /// Getters
    int length() const;
    int initialN() const;
    int initialL() const;
    int initialM() const;

    /// \brief Getter for spin.
    /// \return Spin
    int spin() const;

    /// \brief Setter for spin.
    /// \param s Spin.
    void spin(int s);

    /// \brief Setter for energy.
    /// \param E_in Energy.
    void E(ldouble E_in);

    /// \brief Setter for energy.
    /// \param E_in Energy.
    void setEPython(ldouble E_in);

    /// \brief Getter for energy.
    /// \return Energy.
    ldouble E() const;

    /// \brief Getter for energy in Python.
    /// \return Energy.
    ldouble EPython() const;

    /// \brief Return non-normalised orbital value in a Grid point.
    /// \param i Grid point index.
    /// \param l Value of l
    /// \param m Value of m
    /// \return Non-normalised orbital value.
    ldouble &operator()(int i, int l, int m);
    const ldouble operator()(int i, int l, int m) const;

    /// \brief Get value of normalised orbital in a points.
    /// \param i Grid point index.
    /// \param l Value of l.
    /// \param m Value of m.
    /// \param g Grid object.
    /// \return Orbital value.
    const ldouble getNorm(int i, int l, int m, const Grid &g);

    /// \brief Get normalised orbital in spherical harmonic component (l, m)
    /// \param lo Value of l.
    /// \param mo Value of m.
    /// \return List of orbital values in the Grid points.
    python::list getNormPython(int lo, int mo);

    /// \brief Get normalised orbital assuming central potential approximation.
    /// \return List of orbital values in the Grid points.
    python::list getCentralNormPython();

    /// \brief Normalise orbital
    /// \param g Grid
    void normalise(const Grid &g);

  private:

    /// \brief Load orbital array with set size.
    void load();

    /// Spin
    int _s;

    /// Grid size
    int _N;

    /// Orbital wave function array
    ldouble *_wf;

    /// Quantum number n
    int _initial_n;

    /// Quantum number l
    int _initial_l;

    /// Quantum number m
    int _initial_m;

    /// Energy
    ldouble _E;

    /// List of spherical harmonic components
    std::vector<lm> _sphHarm;

    /// Needs to be renormalised?
    bool _torenorm;

    /// Cached normalised orbital
    ldouble *_wf_norm;
};

#endif
