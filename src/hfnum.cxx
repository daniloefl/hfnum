#include <boost/python.hpp>
#include <string>
#include "HF.h"
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
    .def("E", &Orbital::EPython)
  ;
  class_<HF>("HF", init<object, double>())
    .def(init<object, double>())
    .def("solve", &HF::solve)
    .def("addOrbital", &HF::addOrbitalPython)
    .def("gammaSCF", &HF::gammaSCF)
    .def("method", &HF::method)
    .def("getNucleusPotential", &HF::getNucleusPotentialPython)
    .def("getDirectPotential", &HF::getDirectPotentialPython)
    .def("getExchangePotential", &HF::getExchangePotentialPython)
  ;
}

