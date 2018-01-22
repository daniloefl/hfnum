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

    /// \brief Load basis
    /// \param fname File name
    void loadBasis(const std::string &fname);

  protected:

    /// Basis
    GTO _g;

    /// Fock matrix
    MatrixXld _F;

    /// Overlap matrix
    MatrixXld _S;
};

#endif

