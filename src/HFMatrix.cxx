#include "HF.h"
#include "Grid.h"
#include "Orbital.h"

#include <stdexcept>
#include <exception>

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <cmath>
#include <iostream>
#include <iomanip>

#include "utils.h"

HF::HF(const Grid &g, double Z)
  : _g(g), _Z(Z) {
  _pot.resize(_g.N());
  _vd[std::pair<int, int>(0,  0)] = std::vector<double>(_g.N(), 0);
  _vd[std::pair<int, int>(1, -1)] = std::vector<double>(_g.N(), 0);
  _vd[std::pair<int, int>(1,  0)] = std::vector<double>(_g.N(), 0);
  _vd[std::pair<int, int>(1,  1)] = std::vector<double>(_g.N(), 0);
  _potIndep.resize(_g.N());
  for (int k = 0; k < _g.N(); ++k) {
    _pot[k] = -_Z/_g(k);
    _potIndep[k] = 0;
  }
  _gamma_scf = 0.7;
}

HF::~HF() {
}

std::vector<double> HF::getOrbital(int no, int lo, int mo) {
  Orbital &o = _o[no];
  std::vector<double> res;
  for (int k = 0; k < _g.N(); ++k) {
    res.push_back(o.getNorm(k, lo, mo, _g));
  }
  return res;
}

void HF::gammaSCF(double g) {
  _gamma_scf = g;
}

void HF::solve(int NiterSCF, int Niter, double F0stop) {
  // initialise energies
  for (int k = 0; k < _o.size(); ++k) {
    _o[k].E(-_Z*_Z*0.5/std::pow(_o[k].initialN(), 2)-0.1);
    for (int l = 0; l < _o[k].L()+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        double v = 0;
        if (l == _o[k].initialL() && m == _o[k].initialM()) v = 1;
        for (int ir = 0; ir < _g.N(); ++ir) { // for each radial point
          _o[k](ir, l, m) = v;
        }
      }
    }
  }
  int nStepSCF = 0;
  while (nStepSCF < NiterSCF) {
    std::cout << "SCF step " << nStepSCF << std::endl;
    solveForFixedPotentials(Niter, F0stop);
    nStepSCF++;
    calculateVd(_gamma_scf);
  }
}

// direct potential calculation
// V = int_Oa int_ra rpsi1(ra) rpsi2(ra) Yl1m1(Oa) Yl2m2(Oa)/|ra-rb| ra^2 dOa dra
// with 1 = 2 -> k1
// 1/|ra - rb| = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1) r<^l/r>^(l+1) Y*lm(Oa) Ylm(Ob)
// V = \sum_l=0^inf \sum_m=-l^l ( int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra ) (int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa) Ylm(Ob)
// beta(rb, l) = int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra
// T1 = int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa
// T2 = Ylm(Ob)
//
// We multiply by Y*lm(Ob) and take the existing Ylm(Ob) from the orbital and integrate in dOb to get the radial equations
// int Ylm Y*lm(Ob) Ylm(Ob) dOb = (-1)^m int Ylm Yl(-m) Ylm dOb = sqrt((2l+1)/(4pi)) * CG(l, l, 0, 0, l, 0) * CG(l, l, m, -m, l, -m)
// ---------- T2 = 1/(4pi) int_Ob Ylm(Ob) dOb
//
//
// V = \sum_m \sum_l=0^inf beta(rb, l) T1(l, m) T2(l, m)
//
//
// T1 = int Yl1m1 Yl1m1 Y*lm = (-1)**m int Yl1m1 Yl1m1 Yl(-m)
// T1 = (-1)**m*(-1)**m*np.sqrt((2*l1+1)*(2*l1+1)/(4*np.pi*(2*l+1)))*CG(l1,l1,0,0,l,0)*CG(l1,l1,m1,m1,l,-(-m))
//
// T2 = 1.0/(4*np.pi) int Ylm dOb
//
void HF::calculateVd(double gamma) {
  for (auto &vdLm : _vd) {
    int vdl = vdLm.first.first;
    int vdm = vdLm.first.second;
    std::vector<double> &currentVd = vdLm.second;

    std::vector<double> vd(_g.N(), 0);
    // loop over orbitals
    for (int k1 = 0; k1 < _o.size(); ++k1) {
      for (int l1 = 0; l1 < _o[k1].L()+1; ++l1) {
        for (int m1 = -l1; m1 < l1+1; ++m1) {

          int lmax = 2;
          for (int l = 0; l < lmax+1; ++l) {
            for (int ir2 = 0; ir2 < _g.N(); ++ir2) {
              double beta = 0;
              double r2 = _g(ir2);
              for (int ir1 = 0; ir1 < _g.N(); ++ir1) {
                double r1 = _g(ir1);
                double dr = 0;
                if (ir1 < _g.N()-1) dr = _g(ir1+1) - _g(ir1);
                double rs = r1;
                double rb = r2;
                if (rb < rs) {
                  rs = r2;
                  rb = r1;
                }
                beta += 4*M_PI/(2.0*l + 1.0)*std::pow(_o[k1].getNorm(ir1, l1, m1, _g), 2)*std::pow(rs, l)/std::pow(rb, l+1)*std::pow(r1, 2)*dr;
              }
              double T = 0;
              for (int m = -l; m < l + 1; ++m) {
                double T1 = std::pow(-1, m1)*std::sqrt((2*l1+1)*(2*l1+1)/(4*M_PI*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m));
                double T2 = 0;
                //if (l == vdl && m == -vdm) T2 = 1.0/std::sqrt(4*M_PI);
                // int Ylm Y*lm(Ob) Ylm(Ob) dOb = (-1)^m int Ylm Yl(-m) Ylm dOb = sqrt((2l+1)/(4pi)) * CG(l, l, 0, 0, l, 0) * CG(l, l, m, -m, l, -m)
                if (l == vdl && m == vdm)
                  T2 = std::sqrt((2*l+1)/(4*M_PI))*CG(l, l, 0, 0, l, 0)*CG(l, l, m, -m, l, -m);
                T += T1*T2;
              }
              vd[ir2] += beta*T;
            }
          }

        }
      }
    }
    // for a test in He
    for (int k = 0; k < _g.N(); ++k) vd[k] *= 0.5;

    for (int k = 0; k < _g.N(); ++k) currentVd[k] = (1-gamma)*currentVd[k] + gamma*vd[k];
  }
}

std::vector<double> HF::getNucleusPotential() {
  return _pot;
}

std::vector<double> HF::getDirectPotential() {
  return _vd[std::pair<int, int>(0, 0)];
}

void HF::solveForFixedPotentials(int Niter, double F0stop) {
  int Nr = _g.N();
  int No = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int j = 0; j < _o[k].L()+1; ++j) {
      No += Nr*(2*j + 1);
    }
  }
  int Ne = 0;
  int NL = No;
  for (int k = 0; k < _o.size(); ++k) {
    NL += 1;
    Ne += 1;
  }

  _J.resize(No + Ne + 1, No + Ne + 1);
  _F0.resize(No + Ne + 1, 1);
  _J.setZero();
  _F0.setZero();

  double gamma = 0.2; // move in the direction of the negative slope with this velocity per step
  double newF0Sum = 0;
  for (int k = 0; k < No + Ne + 1; ++k) {
    newF0Sum += std::pow(_F0.coeffRef(k, 0), 2);
  }

  int nStep = 0;
  while (nStep < Niter) {
    // compute sum of squares of F(x_old)
    _nF0 = 0;
    for (int k = 0; k < No + Ne + 1; ++k) {
      _nF0 += std::pow(_F0.coeffRef(k, 0), 2);
    }
    if (nStep >= 1 && _nF0 < F0stop) break;

    nStep += 1;
    step();

    newF0Sum = 0;
    for (int k = 0; k < No + Ne + 1; ++k) {
      newF0Sum += std::pow(_F0.coeffRef(k, 0), 2);
    }

    // limit maximum energy step in a single direction to be 0.1*gamma
    double gscale = 1;

    // change orbital energies
    std::cout << "Orbital energies at step " << nStep << ", ended with minimization constraint = " << std::setw(16) << _nF0 << std::endl;
    std::cout << std::setw(5) << "Index" << " " << std::setw(16) << "Energy (H)" << " " << std::setw(16) << "dE (H) " << std::endl;
    for (int k = 0; k < _o.size(); ++k) {
      std::cout << std::setw(5) << k << " " << std::setw(16) << std::setprecision(14) << _o[k].E() << " " << std::setw(16) << std::setprecision(16) << - gamma*gscale*_dz.coeffRef(No + k, 0) << std::endl;
      _o[k].E(_o[k].E() - gamma*gscale*_dz.coeffRef(No + k, 0));
    }
    // change orbitals
    int idxOrbital = 0;
    for (int k = 0; k < _o.size(); ++k) {
      for (int l = 0; l < _o[k].L()+1; ++l) {
        for (int m = -l; m < l+1; ++m) {
          int idxM = idxOrbital + (l+m)*Nr;
          for (int ir = 0; ir < _g.N(); ++ir) { // for each radial point
            _o[k](ir, l, m) += - gamma*gscale*_dz.coeffRef(idxM + ir, 0);
          }
        }
      }
      idxOrbital += Nr*(2*_o[k].L() + 1);
    }

  }
}

void HF::step() {
  int Nr = _g.N();
  int No = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int j = 0; j < _o[k].L()+1; ++j) {
      No += Nr*(2*j + 1);
    }
  }
  int Ne = 0;
  int NL = No;
  for (int k = 0; k < _o.size(); ++k) {
    NL += 1;
    Ne += 1;
  }
  _J.setZero();
  _F0.setZero();

  prepareKinetic();
  //std::cout << "To solve system with J and F0:" << std::endl;
  //std::cout << _J << std::endl;
  //std::cout << _F0 << std::endl;

  //Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
  solver.compute(_J);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Could not solve system.");
  }
  _dz.resize(No + _o.size() + 1, 1);
  _dz = solver.solve(_F0);
  //std::cout << "Solution: " << _dz << std::endl;

}

void HF::prepareKinetic() {
  int Nr = _g.N();
  int No = 0;
  for (int k = 0; k < _o.size(); ++k) {
    for (int j = 0; j < _o[k].L()+1; ++j) {
      No += Nr*(2*j + 1);
    }
  }
  int Ne = 0;
  int NL = No;
  for (int k = 0; k < _o.size(); ++k) {
    NL += 1;
    Ne += 1;
  }
  //_J.coeffRef(i, j) += 0;
  std::vector<Eigen::Triplet<double> > tripletListJ;
  std::vector<Eigen::Triplet<double> > tripletListF0;
  //tripletList.push_back(Eigen::Triplet<double>(i, j, vij));
  int idxOrbital = 0;
  for (int k = 0; k < _o.size(); ++k) { // for each orbital
    for (int l = 0; l < _o[k].L()+1; ++l) {
      int lsum = 0;
      for (int li = 0; li < l; ++li) {
        lsum += 2*li + 1;
      }
      for (int m = -l; m < l+1; ++m) {
        int idxM = idxOrbital + lsum*Nr + (l+m)*Nr;
        // (12 - 10 f_n) y_n - f_{n-1} y_{n-1} - f_{n+1} y_{n+1} + (s[i+1] + 10.0*s[i] + s[i-1]) = 0
        for (int ir = 0; ir < _g.N(); ++ir) { // for each radial point
          //double v = _o[k](ir, l, m);
          double r = _g(ir);
          double v = _o[k].getNorm(ir, l, m, _g)*std::pow(r, 0.5);
          double a = 2*std::pow(r, 2)*(_o[k].E() - _pot[ir] - _vd[std::pair<int, int>(l, m)][ir]) - std::pow(l + 0.5, 2);
          double f = 1 + a*std::pow(_g.dx(), 2)/12.0;
          //double s_coeff = std::pow(_g.dx(), 2)/12.0*2*std::pow(r, 2);
          double s = std::pow(_g.dx(), 2)/12.0*2*std::pow(r, 2)*_potIndep[ir];
          // index of equation
          tripletListF0.push_back(Eigen::Triplet<double>(idxM + ir, 0, (12 - 10*f)*v + 10.0*s));
          tripletListJ.push_back(Eigen::Triplet<double>(idxM + ir, idxM + ir, (12 - 10*f)));
          tripletListJ.push_back(Eigen::Triplet<double>(idxM + ir, No + k, -10*(2*std::pow(r, 2))*(std::pow(_g.dx(), 2)/12.0)*v));
          // ir-1
          if (ir > 0) {
            //double vp = _o[k](ir-1, l, m);
            double rp = _g(ir-1);
            double vp = _o[k].getNorm(ir-1, l, m, _g)*std::pow(rp, 0.5);
            double ap = 2*std::pow(rp, 2)*(_o[k].E() - _pot[ir-1] - _vd[std::pair<int, int>(l, m)][ir-1]) - std::pow(l + 0.5, 2);
            double fp = 1 + ap*std::pow(_g.dx(), 2)/12.0;
            //double sp_coeff = std::pow(_g.dx(), 2)/12.0*2*std::pow(rp, 2);
            double sp = std::pow(_g.dx(), 2)/12.0*2*std::pow(rp, 2)*_potIndep[ir-1];
            tripletListF0.push_back(Eigen::Triplet<double>(idxM + ir, 0, -fp*vp + sp));
            tripletListJ.push_back(Eigen::Triplet<double>(idxM + ir, idxM + ir - 1, -fp));
            tripletListJ.push_back(Eigen::Triplet<double>(idxM + ir, No + k, -(2*std::pow(rp, 2))*(std::pow(_g.dx(), 2)/12.0)*vp));
          }
          // ir+1
          if (ir < Nr - 1) {
            //double vp = _o[k](ir+1, l, m);
            double rp = _g(ir+1);
            double vp = _o[k].getNorm(ir+1, l, m, _g)*std::pow(rp, 0.5);
            double ap = 2*std::pow(rp, 2)*(_o[k].E() - _pot[ir+1] - _vd[std::pair<int, int>(l, m)][ir+1]) - std::pow(l + 0.5, 2);
            double fp = 1 + ap*std::pow(_g.dx(), 2)/12.0;
            //double sp_coeff = std::pow(_g.dx(), 2)/12.0*2*std::pow(rp, 2);
            double sp = std::pow(_g.dx(), 2)/12.0*2*std::pow(rp, 2)*_potIndep[ir+1];
            tripletListF0.push_back(Eigen::Triplet<double>(idxM + ir, 0, -fp*vp + sp));
            tripletListJ.push_back(Eigen::Triplet<double>(idxM + ir, idxM + ir + 1, -fp));
            tripletListJ.push_back(Eigen::Triplet<double>(idxM + ir, No + k, -(2*std::pow(rp, 2))*(std::pow(_g.dx(), 2)/12.0)*vp));
          }
        }
      }
    }
    // add normalisation = 1 constraint
    // each angular part contributes with (sum psi^2*r^2*dr = 1)
    // the spherical harmonics only contribute for the same l and m, due to the orthogonality conditions
    // lots of spherical harmonics missing
    for (int l = 0; l < _o[k].L()+1; ++l) {
      int lsum = 0;
      for (int li = 0; li < l; ++li) {
        lsum += 2*li + 1;
      }
      for (int m = -l; m < l+1; ++m) {
        int idxM = idxOrbital + lsum*Nr + (l+m)*Nr;
        for (int ir = 0; ir < _g.N(); ++ir) { // for each radial point
          //double v = _o[k](ir, l, m);
          double r = _g(ir);
          double v = _o[k].getNorm(ir, l, m, _g)*std::pow(r, 0.5);
          double dr = 0;
          if (ir < Nr-1) {
            dr = _g(ir+1) - _g(ir);
          }
          tripletListF0.push_back(Eigen::Triplet<double>(No + k, 0, std::pow(v*std::pow(r, -0.5), 2)*std::pow(r, 2)*dr));
          tripletListJ.push_back(Eigen::Triplet<double>(No + k, idxM + ir, 2*v*dr*r));
        }
      }
    }
    tripletListF0.push_back(Eigen::Triplet<double>(No + k, 0, -1.0));
    //tripletListF0.push_back(Eigen::Triplet<double>(NL, 0, 0.0)); // lagrange multiplier for sum E^2 to minimize energy
    tripletListJ.push_back(Eigen::Triplet<double>(NL, No + k, -2*_o[k].E()));

    int lsum = 0;
    for (int li = 0; li < _o[k].L()+1; ++li) {
      lsum += 2*li + 1;
    }
    idxOrbital += Nr*lsum;
  }
  tripletListJ.push_back(Eigen::Triplet<double>(NL, NL, 1.0));

  _J.setFromTriplets(tripletListJ.begin(), tripletListJ.end());
  _F0.setFromTriplets(tripletListF0.begin(), tripletListF0.end());

}

void HF::addOrbital(int L, int initial_n, int initial_l, int initial_m) {
  _o.push_back(Orbital(_g.N(), L, initial_n, initial_l, initial_m));
}


