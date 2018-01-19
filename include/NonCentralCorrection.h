/*
 * \class NonCentralCorrection
 *
 * \ingroup hfnum
 *
 * \brief Takes the result of a previous HF atomic calculation and corrects the assumption that the potential is central used in Vd and Vex.
 */
#ifndef NONCENTRALCORRECTION_H
#define NONCENTRALCORRECTION_H

#include "PerturbativeCorrection.h"
#include <string>
#include <map>
#include <vector>
#include "utils.h"

class NonCentralCorrection : public PerturbativeCorrection {
  public:

    /// \brief Constructor
    NonCentralCorrection();

    /// \brief Destructor
    virtual ~NonCentralCorrection();

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