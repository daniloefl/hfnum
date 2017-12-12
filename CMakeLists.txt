project (hfnum)
cmake_minimum_required(VERSION 3.2)

SET(GCC_EXTRA_COMPILE_FLAGS "-g -std=c++11")
SET(GCC_EXTRA_LINK_FLAGS    "-g -std=c++11")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_EXTRA_COMPILE_FLAGS}" )
SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_EXTRA_LINK_FLAGS}" )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Eigen3)

INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.36.0 REQUIRED COMPONENTS python)
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}) 

include_directories (AFTER include)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doxygen.txt)
    add_custom_target(
      doc
      doxygen ${doxyfile}
    )
endif()


subdirs(src)

