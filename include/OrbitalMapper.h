/*
 * \class OrbitalMapper
 *
 * \ingroup hfnum
 *
 * \brief Class that maps the orbital list to indices used to solve equations.
 */

#ifndef ORBITALMAPPER_H
#define ORBITALMAPPER_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>

class OrbitalMapper {
  public:

    /// \brief Constructor.
    /// \param g Grid object.
    /// \param o List of orbitals.
    OrbitalMapper(const Grid &g, std::vector<Orbital *> &o);

    /// \brief Destructor
    virtual ~OrbitalMapper();

    /// \brief Convert orbital index, quantum number l and m and the grid index i into a general index
    /// \param k Orbital index.
    /// \param l l.
    /// \param m m.
    /// \param i Grid index.
    int sparseIndex(int k, int l, int m, int i);

    /// \brief Getter for number of indices for a sparse matrix
    /// \return Number of indices for a sparse matrix.
    int sparseN();

    /// \brief Convert orbital index, quantum number l and m in a general index, assuming no indices are needed for the Grid
    /// \param k Orbital index.
    /// \param l l.
    /// \param m m.
    /// \return Index.
    int index(int k, int l, int m);

    /// \brief Number of non-sparse indices
    /// \return Number of indices
    int N();

    /// \brief Get orbital index from general index
    /// \param idx Index in matrix.
    /// \return Orbital index.
    int orbital(int idx);

    /// \brief Get quantum number l from general index
    /// \param idx Index in matrix.
    /// \return Quantum number l.
    int l(int idx);

    /// \brief Get quantum number m from general index
    /// \param idx Index in matrix.
    /// \return Quantum number m.
    int m(int idx);
    

  private:
    /// Grid
    const Grid &_g;
    /// Orbital list.
    std::vector<Orbital *> &_o;
};

#endif

