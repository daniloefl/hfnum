PYTHON_ADD_MODULE(hfnum utils.cxx Grid.cxx Orbital.cxx OrbitalMapper.cxx LinearSystemBuilder.cxx IterativeStandardSolver.cxx SCF.cxx HF.cxx HFS.cxx DFT.cxx StateReader.cxx PerturbativeCorrection.cxx NonCentralCorrection.cxx SpinOrbitCorrection.cxx Basis.cxx GTO.cxx STO.cxx MatrixSCF.cxx RHF.cxx hfnum.cxx)
target_link_libraries(hfnum ${Boost_LIBRARIES} boost_python)

