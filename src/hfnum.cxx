#include <boost/python.hpp>
#include <string>
#include "HF.h"
#include "HFS.h"
#include "DFT.h"
#include "Grid.h"
#include "Orbital.h"
#include "NonCentralCorrection.h"
#include "SpinOrbitCorrection.h"
#include "RHF.h"
#include "utils.h"
#include "units.h"

#include "HFException.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(hfnum)
{

  scope().attr("eV") = electronVolt; 
  scope().attr("electronVolt") = electronVolt; 

  scope().attr("H") = Hartree;
  scope().attr("Hartree") = Hartree;

  scope().attr("Rd") = Rydberg;
  scope().attr("Rydberg") = Rydberg;


  def("getSymbol", getSymbol);
  class_<HFException>("HFException", init<>())
    .def(init<const std::string>())
    .def("what", &HFException::what)
  ;
  class_<HFS>("HFS", init<>())
    .def(init<>())
    .def(init<ldouble>())
    .def(init<const std::string>())
    .def("solve", &HFS::solve)
    .def("addOrbital", &HFS::addOrbitalPython)
    .def("gammaSCF", &HFS::gammaSCF)
    .def("method", &HFS::method)
    .def("resetGrid", &HFS::resetGrid)
    .def("Z", &HFS::Z)
    .def("setZ", &HFS::setZ)
    .def("getR", &HFS::getR)
    .def("getNucleusPotential", &HFS::getNucleusPotentialPython)
    .def("getDirectPotential", &HFS::getDirectPotentialPython)
    .def("getExchangePotential", &HFS::getExchangePotentialPython)
    .def("getNOrbitals", &HFS::getNOrbitals)
    .def("getOrbital_n", &HFS::getOrbital_n)
    .def("getOrbital_l", &HFS::getOrbital_l)
    .def("getOrbital_m", &HFS::getOrbital_m)
    .def("getOrbital_s", &HFS::getOrbital_s)
    .def("getOrbital_E", &HFS::getOrbital_E)
    .def("getOrbitalName", &HFS::getOrbitalName)
    .def("getCentral", &HFS::getOrbitalCentralPython)
    .def("getE0", &HFS::getE0)
    .def("save", &HFS::save)
    .def("load", &HFS::load)
  ;
  //class_<Grid>("Grid", init<bool, double, int, double>())
  //  .def(init<bool, double, int, double>())
  //  .def("getR", &Grid::getR)
  //;
  class_<RHF>("RHF", init<>())
    .def(init<>())
    .def("solve", &RHF::solve)
    .def("setZ", &RHF::setZ)
    .def("Z", &RHF::Z)
    .def("addOrbital", &RHF::addOrbital)
    .def("getNOrbitals", &RHF::getNOrbitals)
    .def("getOrbital_n", &RHF::getOrbital_n)
    .def("getOrbital_l", &RHF::getOrbital_l)
    .def("getOrbital_m", &RHF::getOrbital_m)
    .def("getOrbital_s", &RHF::getOrbital_s)
    .def("getOrbital_E", &RHF::getOrbital_E)
    .def("getOrbitalName", &RHF::getOrbitalName)
    .def("gammaSCF", &RHF::gammaSCF)
    .def("Nscf", &RHF::Nscf)
    .def("loadBasis", &RHF::loadBasis)
    .def("getOrbital", &RHF::getOrbital)
  ;
  class_<Orbital>("Orbital", init<int, int, const std::string>())
    .def(init<int, int, const std::string>())
    //.def("addSphHarm", &Orbital::addSphHarm)
    //.def("get", &Orbital::getNormPython)
    .def("getCentral", &Orbital::getCentralNormPython)
    .def("E", &Orbital::EPython)
    .def("setE", &Orbital::setEPython)
  ;
  class_<HF>("HF", init<>())
    .def(init<>())
    .def(init<ldouble>())
    .def(init<const std::string>())
    .def("solve", &HF::solve)
    .def("addOrbital", &HF::addOrbitalPython)
    .def("gammaSCF", &HF::gammaSCF)
    .def("method", &HF::method)
    .def("resetGrid", &HF::resetGrid)
    .def("Z", &HF::Z)
    .def("setZ", &HF::setZ)
    .def("getR", &HF::getR)
    .def("getNucleusPotential", &HF::getNucleusPotentialPython)
    .def("getDirectPotential", &HF::getDirectPotentialPython)
    .def("getExchangePotential", &HF::getExchangePotentialPython)
    .def("getNOrbitals", &HF::getNOrbitals)
    .def("getOrbital_n", &HF::getOrbital_n)
    .def("getOrbital_l", &HF::getOrbital_l)
    .def("getOrbital_m", &HF::getOrbital_m)
    .def("getOrbital_s", &HF::getOrbital_s)
    .def("getOrbital_E", &HF::getOrbital_E)
    .def("getOrbitalName", &HF::getOrbitalName)
    .def("getCentral", &HF::getOrbitalCentralPython)
    .def("getE0", &HF::getE0)
    .def("save", &HF::save)
    .def("load", &HF::load)
  ;
  class_<NonCentralCorrection>("NonCentralCorrection", init<>())
    .def(init<>())
    .def("Z", &NonCentralCorrection::Z)
    .def("getR", &NonCentralCorrection::getR)
    .def("correct", &NonCentralCorrection::correct)
    .def("getNOrbitals", &NonCentralCorrection::getNOrbitals)
    .def("getOrbital_n", &NonCentralCorrection::getOrbital_n)
    .def("getOrbital_l", &NonCentralCorrection::getOrbital_l)
    .def("getOrbital_m", &NonCentralCorrection::getOrbital_m)
    .def("getOrbital_s", &NonCentralCorrection::getOrbital_s)
    .def("getOrbital_E", &NonCentralCorrection::getOrbital_E)
    .def("getOrbitalName", &NonCentralCorrection::getOrbitalName)
    .def("getCentral", &NonCentralCorrection::getOrbitalCentralPython)
    .def("getCorrectedE", &NonCentralCorrection::getCorrectedEPython)
    .def("getCoefficients", &NonCentralCorrection::getCoefficientsPython)
    .def("getE0", &NonCentralCorrection::getE0)
    .def("getE0Uncorrected", &NonCentralCorrection::getE0Uncorrected)
    .def("load", &NonCentralCorrection::load)
  ;
  class_<SpinOrbitCorrection>("SpinOrbitCorrection", init<>())
    .def(init<>())
    .def("Z", &SpinOrbitCorrection::Z)
    .def("getR", &SpinOrbitCorrection::getR)
    .def("correct", &SpinOrbitCorrection::correct)
    .def("getNOrbitals", &SpinOrbitCorrection::getNOrbitals)
    .def("getOrbital_n", &SpinOrbitCorrection::getOrbital_n)
    .def("getOrbital_l", &SpinOrbitCorrection::getOrbital_l)
    .def("getOrbital_m", &SpinOrbitCorrection::getOrbital_m)
    .def("getOrbital_s", &SpinOrbitCorrection::getOrbital_s)
    .def("getOrbital_E", &SpinOrbitCorrection::getOrbital_E)
    .def("getOrbitalName", &SpinOrbitCorrection::getOrbitalName)
    .def("getCentral", &SpinOrbitCorrection::getOrbitalCentralPython)
    .def("getCorrectedE", &SpinOrbitCorrection::getCorrectedEPython)
    .def("getCoefficients", &SpinOrbitCorrection::getCoefficientsPython)
    .def("getE0", &SpinOrbitCorrection::getE0)
    .def("getE0Uncorrected", &SpinOrbitCorrection::getE0Uncorrected)
    .def("load", &SpinOrbitCorrection::load)
  ;
  class_<DFT>("DFT", init<>())
    .def(init<>())
    .def(init<ldouble>())
    .def(init<const std::string>())
    .def("solve", &DFT::solve)
    .def("addOrbital", &DFT::addOrbitalPython)
    .def("gammaSCF", &DFT::gammaSCF)
    .def("method", &DFT::method)
    .def("resetGrid", &DFT::resetGrid)
    .def("Z", &DFT::Z)
    .def("setZ", &DFT::setZ)
    .def("getR", &DFT::getR)
    .def("getDensityUp", &DFT::getDensityUpPython)
    .def("getDensityDown", &DFT::getDensityDownPython)
    .def("getHartree", &DFT::getHartreePython)
    .def("getExchangeUp", &DFT::getExchangeUpPython)
    .def("getExchangeDown", &DFT::getExchangeDownPython)
    .def("getNucleusPotential", &DFT::getNucleusPotentialPython)
    .def("getNOrbitals", &DFT::getNOrbitals)
    .def("getOrbital_n", &DFT::getOrbital_n)
    .def("getOrbital_l", &DFT::getOrbital_l)
    .def("getOrbital_m", &DFT::getOrbital_m)
    .def("getOrbital_s", &DFT::getOrbital_s)
    .def("getOrbital_E", &DFT::getOrbital_E)
    .def("getOrbitalName", &DFT::getOrbitalName)
    .def("getCentral", &DFT::getOrbitalCentralPython)
    .def("getE0", &DFT::getE0)
    .def("save", &DFT::save)
    .def("load", &DFT::load)
  ;
}

