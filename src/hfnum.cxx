#include <boost/python.hpp>
#include <string>
#include "HFPy.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(hfnum)
{
  class_<HFPy>("HF", init<double, int, double, double>())
    .def(init<double, int, double, double>())
    .def("solve", &HFPy::solve)
    .def("addOrbital", &HFPy::addOrbital)
    .def("getOrbital", &HFPy::getOrbital)
    .def("getR", &HFPy::getR)
    .def("gammaSCF", &HFPy::gammaSCF)
    .def("getNucleusPotential", &HFPy::getNucleusPotential)
    .def("getDirectPotential", &HFPy::getDirectPotential)
    .def("getExchangePotential", &HFPy::getExchangePotential)
  ;
}

