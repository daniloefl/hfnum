PYTHON_ADD_MODULE(hfnum utils.cxx Grid.cxx Orbital.cxx OrbitalMapper.cxx LinearSystemBuilder.cxx IterativeGordonSolver.cxx IterativeRenormalisedSolver.cxx SCF.cxx HF.cxx DFT.cxx hfnum.cxx)
target_link_libraries(hfnum ${Boost_LIBRARIES} boost_python)

