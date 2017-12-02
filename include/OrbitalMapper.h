#ifndef ORBITALMAPPER_H
#define ORBITALMAPPER_H

#include "Grid.h"
#include "Orbital.h"
#include <vector>

class OrbitalMapper {
  public:
    OrbitalMapper(const Grid &g, std::vector<Orbital> &o);
    virtual ~OrbitalMapper();

    // convert orbital index, quantum number l and m and the grid index i into a general index
    int sparseIndex(int k, int l, int m, int i);

    // number of indices for a sparse matrix
    int sparseN();

    // convert orbital index, quantum number l and m in a general index, assuming no indices are needed for the Grid
    int index(int k, int l, int m);

    // number of indices
    int N();

    // get orbital index from general index
    int orbital(int idx);

    // get quantum number l from general index
    int l(int idx);

    // get quantum number m from general index
    int m(int idx);
    

  private:
    const Grid &_g;
    std::vector<Orbital> &_o;
};

#endif

