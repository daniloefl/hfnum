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

#include "utils.h"

class Orbital {
  public:

    /// \brief Constructor.
    /// \param s Spin.
    /// \param n Quantum number n. For initial conditions.
    /// \param l Quantum number l. For initial conditions.
    /// \param m Quantum number m. For initial conditions.
    Orbital(int s, int n, int l, int m);

    /// \brief Constructor.
    /// \param n Quantum number n. For initial conditions.
    /// \param l Quantum number l. For initial conditions.
    /// \param term Content to be specified in the format "+-+-  ", indicating + or - for filled electrons and " " for non-filled ones. The first characters refer to ml = -l and the last ones to ml = +l..
    Orbital(int n, int l, const std::string term = "");

    /// \brief Constructor.
    /// \param o Object to copy.
    Orbital(const Orbital &o);

    /// \brief Destructor.
    virtual ~Orbital();

    /// \brief Assignment operator.
    /// \param o Orbital to copy.
    /// \return This.
    Orbital &operator =(const Orbital &o);

    /// \brief Set number of Grid points.
    /// \param N number of Grid points.
    void N(int N);

    /// \brief Getter for number of Grid points.
    /// \return Number of Grid points.
    int N() const;

    /// Getters
    int length() const;
    int n() const;
    int l() const;
    int m() const;
    int g() const;
    const std::string &term() const;

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
    /// \return Non-normalised orbital value.
    ldouble &operator()(int i);
    const ldouble operator()(int i) const;

    /// \brief Get value of normalised orbital in a points.
    /// \param i Grid point index.
    /// \param g Grid object.
    /// \return Orbital value.
    const ldouble getNorm(int i, const Grid &g);

    /// \brief Get normalised orbital.
    /// \return List of orbital values in the Grid points.
    python::list getNormPython();

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
    int _n;

    /// Quantum number l
    int _l;

    /// Quantum number m
    int _m;

    /// Multiplicity
    ldouble _g;

    /// term name
    std::string _term;

    /// Energy
    ldouble _E;

    /// Needs to be renormalised?
    bool _torenorm;

    /// Cached normalised orbital
    ldouble *_wf_norm;
};

#endif
