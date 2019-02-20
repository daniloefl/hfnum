ADD_LIBRARY(hfnum MODULE utils.cxx Grid.cxx Orbital.cxx OrbitalMapper.cxx LinearSystemBuilder.cxx IterativeStandardSolver.cxx SCF.cxx HF.cxx HFS.cxx DFT.cxx StateReader.cxx PerturbativeCorrection.cxx NonCentralCorrection.cxx SpinOrbitCorrection.cxx Basis.cxx GTO.cxx STO.cxx MatrixSCF.cxx RHF.cxx HFException.cxx hfnum.cxx)
target_link_libraries(hfnum ${Boost_LIBRARIES} boost_python GSL::gsl ${Python3_LIBRARIES})
set_property (TARGET hfnum PROPERTY PREFIX "")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set_property (TARGET hfnum PROPERTY SUFFIX ".pyd")
endif()

