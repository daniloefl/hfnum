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

class Orbital {
  public:
    Orbital(int s = 1, int initial_n = 1, int initial_l = 0, int initial_m = 0);
    Orbital(const Orbital &o);
    virtual ~Orbital();

    Orbital &operator =(const Orbital &o);

    void addSphHarm(int l, int m);
    const std::vector<lm> &getSphHarm() const;

    void N(int N);
    int N() const;
    int length() const;

    int initialN() const;
    int initialL() const;
    int initialM() const;

    int spin() const;
    void spin(int s);

    void E(double E_in);
    double E() const;

    double &operator()(int i, int l, int m);
    const double operator()(int i, int l, int m) const;
    const double getNorm(int i, int l, int m, const Grid &g);
    python::list getNormPython(int lo, int mo);
    void normalise(const Grid &g);

  private:

    void load();

    int _s;

    int _N;
    double *_wf;

    int _initial_n;
    int _initial_l;
    int _initial_m;

    double _E;

    std::vector<lm> _sphHarm;

    bool _torenorm;
    double *_wf_norm;
};

#endif
