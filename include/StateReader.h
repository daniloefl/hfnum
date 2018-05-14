/*
 * \class StateReader
 *
 * \ingroup hfnum
 *
 * \brief Reads a saved file containing the state of the SCF calculation.
 */
#ifndef STATEREADER_H
#define STATEREADER_H

#include <fstream>
#include <string>
#include <vector>
#include "Orbital.h"
#include "Grid.h"
#include "utils.h"

class StateReader {
  public:

    /// \brief Constructor
    StateReader();

    /// \brief Constructor that automatically loads a file
    /// \param fin Input file name
    StateReader(const std::string &fin);

    /// \brief Destructor
    virtual ~StateReader();

    /// \brief Load an input file
    /// \param fin Input file name
    void load(const std::string &fin);

    /// \brief Getter function for vector of doubles
    /// \param id Name of the parameter
    /// \return Vector of doubles written under that name
    std::vector<ldouble> &getVector(const std::string &id);

    /// \brief Getter function for vector of doubles identified by a single number
    /// \param i Key
    /// \return Vector of doubles written under a single id key
    std::vector<ldouble> &getVd(const int i);

    /// \brief Getter function for vector of doubles identified by a pair of numbers
    /// \param i Parameter 1
    /// \param j Parameter 2
    /// \return Vector of doubles written under a pair of keys
    std::vector<ldouble> &getVex(const int i, const int j);

    /// \brief Getter function for Orbital
    /// \param id Name of the parameter
    /// \return Orbital written under that number
    Orbital *getOrbital(const int i);

    /// \brief Getter function for doubles
    /// \param id Name of the parameter
    /// \return Double written under that name
    ldouble getDouble(const std::string &id);

    /// \brief Getter function for int
    /// \param id Name of the parameter
    /// \return Integer written under that name
    int getInt(const std::string &id);

    /// Commonly used saved parameters
    std::vector<Orbital *> _o;
    std::map<int, std::vector<ldouble> > _vd;
    std::map<std::pair<int,int>, std::vector<ldouble> > _vex;

    std::map<int, int> _lambdaMap;
    std::vector<ldouble> _lambda;

  protected:

    /// Saved loaded parameters
    std::map<std::string, std::vector<ldouble> > _v;
    std::map<std::string, ldouble> _d;
    std::map<std::string, int> _i;

};

#endif
