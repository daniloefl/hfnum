#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <complex>

using namespace Eigen;

/// (l,m) pair
class lm {
  public:
  int l;
  int m;
  lm(int li = 0, int mi = 0);
  ~lm();
  lm(const lm &o);
  lm &operator =(const lm &o);
  bool operator ==(const lm &o) const;
  bool operator !=(const lm &o) const;
  bool operator >(const lm &o) const;
  bool operator <(const lm &o) const;
  std::ostream &operator <<(std::ostream &os) const;
};

/// Scalar type. Used to have a single long double type everywhere.
typedef long double ldouble;
typedef std::complex<ldouble> cldouble;

/// Dynamic dense matrix
typedef Matrix<ldouble, Dynamic, Dynamic> MatrixXld;
typedef Matrix<cldouble, Dynamic, Dynamic> MatrixXcld;

/// Dynamic dense column vector.
typedef Matrix<ldouble, Dynamic, 1> VectorXld;
typedef Matrix<cldouble, Dynamic, 1> VectorXcld;

/// Types used for direct and exchange potential storage.
typedef std::vector<ldouble> Vradial;

/// Sparse matrix
typedef SparseMatrix<ldouble, ColMajor> SMatrixXld;

/// Sparse vector
typedef SparseMatrix<ldouble, ColMajor> SVectorXld;

/// Triplet used to fast insertion in sparse matrix
typedef Triplet<ldouble> Tr;

/// Factorial calculation
double factorial(double n);

/// Clebsch-Gordon coefficients
double CG(int j1, int j2, int m1, int m2, int j, int m);

#endif

