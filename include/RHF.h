/*
 * \class RHF
 *
 * \ingroup hfnum
 *
 * \brief Implements the Roothan-HF eq.
 */

#ifndef RHF_H
#define RHF_H

#include <vector>
#include <map>

#include "utils.h"
#include "Basis.h"
#include "MatrixSCF.h"
#include "GTO.h"

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <boost/python/list.hpp>

class RHF : public MatrixSCF {
  public:

    /// \brief Constructor for an atom.
    RHF();

    /// \brief Destructor.
    virtual ~RHF();

    /// \brief Solve Roothan-Hartree-Fock equation
    void solveRoothan();

    /// \brief Recalculate SCF and call solveRoothan iteratively
    void solve();

    /// \brief Set Z.
    /// \param Z Z value.
    void setZ(ldouble Z);

    /// \brief Get orbital as a function of r
    /// \param no Orbital number
    /// \param s Spin up or down
    /// \param l_proj Orbital angular momentum l to project on
    /// \param m_proj Orbital angular momentum m to project on
    /// \param r List of r values
    /// \return Orbital values
    boost::python::list getOrbital(int no, int s, int l_proj, int m_proj, boost::python::list r);

    /// \brief Load basis
    /// \param fname File name
    void loadBasis(const std::string &fname);

  protected:

    /// Basis
    GTO _g;

    /// Fock matrix
    MatrixXld _F_up;
    MatrixXld _F_dw;

    /// Overlap matrix
    MatrixXld _S;

    /// old coeff. matrix
    MatrixXld _old_c_up;
    MatrixXld _old_c_dw;

    /// coeff. matrix
    MatrixXld _c_up;
    MatrixXld _c_dw;

    /// filled orbitals
    int _Nfilled_up;
    int _Nfilled_dw;
};

#endif

