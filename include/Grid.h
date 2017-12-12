/**
 * \class Grid
 *
 * \ingroup hfnum
 *
 * \brief Represents a grid in the radial distance to the nucleus. Can be a logarithmic Grid or a linear Grid.
 *
 */


#ifndef GRID_H
#define GRID_H

#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <Python.h>
using namespace boost;

class Grid {
  public:

    /// \brief Constructor with Grid parameters.
    /// \param isLog Whether the Grid is logarithmic. If this is true, the Grid is r = exp(rmin + dx*i), otherwise: r = rmin + dx*i for i = 0..N-1
    /// \param dx Step size.
    /// \param N Number of Grid points.
    /// \param rmin Minimum Grid point.
    Grid(bool isLog = true, double dx = 1e-1, int N = 150, double rmin = 1e-4);

    /// \brief Copy constructor.
    /// \param g Other Grid object.
    Grid(const Grid &g);

    /// \brief Assignment operator.
    /// \param g Other Grid object.
    Grid &operator =(const Grid &g);

    /// \brief Destructor.
    virtual ~Grid();

    /// \brief Getter for dx.
    double dx() const;

    /// \brief Getter for N.
    int N() const;

    /// \brief Getter for position.
    /// \param i Index.
    double operator()(int i) const;

    /// \brief Getter for isLog flag.
    bool isLog() const;

    /// \brief Python interface for getting list of radial values.
    python::list getR() const;

  private:

    /// Grid values
    double *_r;

    /// Step size
    double _dx;

    /// Number of grid points
    int _N;

    /// Minimum r value
    double _rmin;

    /// Whether the grid is logarithmic
    bool _isLog;
};

#endif

