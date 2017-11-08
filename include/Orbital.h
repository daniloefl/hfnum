#ifndef ORBITAL_H
#define ORBITAL_H

class Grid;

class Orbital {
  public:
    Orbital(int N = 150, int L = 1, int initial_n = 1, int initial_l = 0, int initial_m = 0);
    Orbital(const Orbital &o);
    virtual ~Orbital();

    Orbital &operator =(const Orbital &o);

    void N(int N);
    void L(int L);
    int N() const;
    int L() const;
    int length() const;

    int initialN() const;
    int initialL() const;
    int initialM() const;

    void E(double E_in);
    double E() const;

    double &operator()(int i, int l, int m);
    const double operator()(int i, int l, int m) const;
    const double getNorm(int i, int l, int m, const Grid &g);

  private:

    void load();

    int _N;
    int _L;
    double *_wf;

    int _initial_n;
    int _initial_l;
    int _initial_m;

    double _E;

    bool _torenorm;
    double *_wf_norm;
};

#endif
