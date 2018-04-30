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

    /// \brief Convert orbital index and the grid index i into a general index
    /// \param k Orbital index.
    /// \param i Grid index.
    int sparseIndex(int k, int i);

    /// \brief Getter for number of indices for a sparse matrix
    /// \return Number of indices for a sparse matrix.
    int sparseN();

    /// \brief Convert orbital index in a general index, assuming no indices are needed for the Grid
    /// \param k Orbital index.
    /// \return Index.
    int index(int k);

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

    /// \brief Get quantum number s from general index
    /// \param idx Index in matrix.
    /// \return Quantum number s.
    int s(int idx);
    

  private:
    /// Grid
    const Grid &_g;
    /// Orbital list.
    std::vector<Orbital *> &_o;
};

#endif

