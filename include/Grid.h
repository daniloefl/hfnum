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

enum gridType {
  linGrid = 0,
  expGrid = 1,
  linExpGrid = 2
};

class Grid {
  public:

    /// \brief Constructor with Grid parameters.
    /// \param t Whether the Grid is linear, exponential or linear + exponential. If this is set to expGrid, the Grid is r = exp(rmin + dx*i). If it is set to linGrid, it is r = rmin + dx*i for i = 0..N-1. If it is set to linExpGrid, the grid is such that rmin + i*dx = rmin*r + alpha*log(r)
    /// \param dx Step size.
    /// \param N Number of Grid points.
    /// \param rmin Minimum Grid point.
    Grid(gridType t = expGrid, double dx = 1e-1, int N = 150, double rmin = 1e-4, double beta = 0, double Z = 1);

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

    /// \brief Getter for isLinExp flag.
    bool isLinExp() const;

    /// \brief Getter for type.
    gridType type() const;

    /// \brief Python interface for getting list of radial values.
    python::list getR() const;

    /// \brief Reset configuration.
    /// \param t Grid type.
    /// \param dx Step size.
    /// \param N Number of Grid points.
    /// \param rmin Minimum Grid point.
    /// \param beta Parameter for the lin+exp grid.
    /// \param Z Atomic number
    void reset(gridType t = expGrid, double dx = 1e-1, int N = 150, double rmin = 1e-4, double beta = 0.0, double Z = 1.0);


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
    gridType _t;

    /// alpha, beta, Z
    double _alpha;
    double _beta;
    double _Z;
    double _x0;
};

#endif

