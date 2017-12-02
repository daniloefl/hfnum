#ifndef HFPY_H
#define HFPY_H

#include "Grid.h"
#include "HF.h"
#include "Orbital.h"
#include <boost/range/irange.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <vector>
#include <Python.h>

class HFPy {

  public:
    HFPy(bool isLog = true, double dx = 1e-1, int N = 150, double rmin = 1e-4, double Z = 1.0);
    virtual ~HFPy();

    void solve(int NiterSCF, int Niter, double F0stop);
    void addOrbital(int L, int s, int initial_n, int initial_l, int initial_m);
    boost::python::list getR() const;
    boost::python::list getOrbital(int no, int mo, int lo);
    void gammaSCF(double g);
    void method(int m);

    boost::python::list getNucleusPotential();
    boost::python::list getDirectPotential(int k);
    boost::python::list getExchangePotential(int k, int k2);

  private:
    Grid _g;
    HF _h;

};

#endif

