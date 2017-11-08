PYTHON_ADD_MODULE(hfnum utils.cxx Grid.cxx Orbital.cxx HF.cxx HFPy.cxx hfnum.cxx)
target_link_libraries(hfnum ${Boost_LIBRARIES} boost_python)

