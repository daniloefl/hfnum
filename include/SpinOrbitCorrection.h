/*
 * \class SpinOrbitCorrection
 *
 * \ingroup hfnum
 *
 * \brief Takes the result of a previous HF atomic calculation and corrects for missing spin-orbit coupling.
 */
#ifndef SPINORBITCORRECTION_H
#define SPINORBITCORRECTION_H

#include "PerturbativeCorrection.h"
#include <string>
#include <map>
#include <vector>
#include "utils.h"

class SpinOrbitCorrection : public PerturbativeCorrection {
  public:

    /// \brief Constructor
    SpinOrbitCorrection();

    /// \brief Destructor
    virtual ~SpinOrbitCorrection();

    /// \brief Load input file with previously saved state
    /// \param fname Input file name
    void load(const std::string &fname);

    /// \brief Function to be implemented to calculate perturbative correction
    void correct();
    
  protected:
    std::map<int, std::vector<ldouble> > _vd;
    std::map<std::pair<int,int> , std::vector<ldouble> > _vex;
};

#endif
