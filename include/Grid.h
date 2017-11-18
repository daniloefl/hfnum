#ifndef GRID_H
#define GRID_H

class Grid {
  public:
    Grid(bool isLog = true, double dx = 1e-1, int N = 150, double rmin = 1e-4);
    Grid(const Grid &g);
    Grid &operator =(const Grid &g);
    virtual ~Grid();

    double dx() const;
    int N() const;

    double operator()(int i) const;
    bool isLog() const;

  private:
    double *_r;
    double _dx;
    int _N;
    double _rmin;
    bool _isLog;
};

#endif

