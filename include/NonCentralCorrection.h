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

    /// \brief Get corrected full atom energy level
    /// \return Corrected energy level
    ldouble getE0();

    /// \brief Get original full atom energy level
    /// \return Original energy level
    ldouble getE0Uncorrected();
    
  protected:
    std::map<int, std::vector<ldouble> > _vd;
    std::map<std::pair<int,int> , std::vector<ldouble> > _vex;

    MatrixXld _J;
    MatrixXld _K;

    MatrixXld _Jcorr;
    MatrixXld _Kcorr;
};

#endif
