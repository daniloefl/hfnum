#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;

typedef long double ldouble;
typedef Matrix<ldouble, Dynamic, Dynamic> MatrixXld;
typedef Matrix<ldouble, Dynamic, 1> VectorXld;

typedef std::map<std::pair<int, int>, std::vector<ldouble> > Vd;
typedef std::map<std::pair<int, int>, std::vector<ldouble> > Vex;

typedef SparseMatrix<ldouble, ColMajor> SMatrixXld;
typedef SparseMatrix<ldouble, ColMajor> SVectorXld;
typedef Triplet<ldouble> Tr;

double factorial(double n);
double CG(int j1, int j2, int m1, int m2, int j, int m);

#endif

