#include <boost/python.hpp>
#include <string>
#include "HF.h"
#include "DFT.h"
#include "Grid.h"
#include "Orbital.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(hfnum)
{
  class_<Grid>("Grid", init<bool, double, int, double>())
    .def(init<bool, double, int, double>())
    .def("getR", &Grid::getR)
  ;
  class_<Orbital>("Orbital", init<int, int, int, int>())
    .def(init<int, int, int, int>())
    .def("addSphHarm", &Orbital::addSphHarm)
    .def("get", &Orbital::getNormPython)
    .def("getCentral", &Orbital::getCentralNormPython)
    .def("E", &Orbital::EPython)
    .def("setE", &Orbital::setEPython)
  ;
  class_<HF>("HF", init<>())
    .def(init<>())
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
  class_<DFT>("DFT", init<>())
    .def(init<>())
    .def(init<const std::string>())
    .def("solve", &DFT::solve)
    .def("addOrbital", &DFT::addOrbitalPython)
    .def("gammaSCF", &DFT::gammaSCF)
    .def("method", &DFT::method)
    .def("resetGrid", &DFT::resetGrid)
    .def("Z", &DFT::Z)
    .def("setZ", &DFT::setZ)
    .def("getR", &DFT::getR)
    .def("getNucleusPotential", &DFT::getNucleusPotentialPython)
    .def("getDirectPotential", &DFT::getDirectPotentialPython)
    .def("getExchangePotential", &DFT::getExchangePotentialPython)
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

