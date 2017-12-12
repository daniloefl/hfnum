#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;

/// Scalar type. Used to have a single long double type everywhere.
typedef long double ldouble;

/// Dynamic dense matrix
typedef Matrix<ldouble, Dynamic, Dynamic> MatrixXld;

/// Dynamic dense column vector.
typedef Matrix<ldouble, Dynamic, 1> VectorXld;

/// Types used for direct and exchange potential storage.
typedef std::map<std::pair<int, int>, std::vector<ldouble> > Vd;
typedef std::map<std::pair<int, int>, std::vector<ldouble> > Vex;

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

