# hfnum

Hartree-Fock calculation in C++ using a numerical Grid. Based on hfpython repository.
It currently can use linear or logarithmic Grids, but only logarithmic Grids have been observed to work with acceptable precision.

Three methods are available to solve the differential equation:
  * method 0: Sparse Numerov Matrix method
    * Creates one numerical equation per differential equation and Grid point and puts them all in an NxN sparse matrix. Extra equations are created to force the normalisation of the eigenfunctions to be 1. Since the normalisation condition is non-linear, the system is resolved using the Newton-Raphson method, by calculating the Jacobian matrix of partial derivatives and changing the energy and function values according to -X inverse(Jacobian), where X is the column-vector of wavefunction values and energies. This method is extremely slow, but it is simple and assumes only that the wave function first and last values are zero.
  * method 1: Iterative Numerov Method using Gordon's method to guess initial conditions
    * The system can be solved (up to a normalisation) using Numerov equation to get the third point based on the two points before it. However, we need two initial conditions and choosing the wrong initial conditions (particularly in non-symmetric systems, such as atoms with more than 2 electrons) can lead to divergence. Gordon's solution tries a set of linearly independent solutions and uses a clever method to discover the correct initial conditions. It is described here: http://aip.scitation.org/doi/pdf/10.1063/1.436421
  * method 2: Iterative renormalised method
    * This method is an extension of the method proposed by Gordon. The method is only re-written in a different way using the ratio of solutions normalised by the differential equation coefficients. This procedure avoids overflows, which happen in method 1. It is recommended and it is explained here: http://aip.scitation.org/doi/pdf/10.1063/1.436421

The software is a Python library, where the calculations are done in C++, but the configuration of the parameters is done in Python.
Example Python configurations for the Hydrogen, Helium, Lithium and Beryllium can be seen in the share directory.
A configuration for Borum is available, but there is a bug if l > 0, preventing the code from working properly in this case.

The basic configuration works as follows:

```
# add path of the library hfnum.so in the PYTHONPATH or use the following:
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

# for some calculations
import numpy as np

# this is the library we need
import hfnum

# atomic number
Z = 3

# log grid
# r = exp(log(rmin) + dx * i), where i = 0..N-1
# change the Grid parameters below
dx = 1e-1/Z       # Grid step
N = 255*Z         # number of points
rmin = 1e-10      # first point in the Grid

# this is the main solver
h = hfnum.HF()

# initialise library with the Grid parameters
# the first parameter tells it whether one should use the logarithmic Grid
# the linear Grid works poorly, so it is recommended to keep this always in True
h.resetGrid(True, dx, int(N), rmin)

# set atomic number
h.setZ(Z)

# use this to use the faster method, which iteratively looks for the energy
# without solving the equations using the NxN grid of points
h.method(2)

# create an Orbital as many times as needed
# the syntax is the following:
# myVar = hfnum.Orbital(spin, n l, m)
# n, l and m are the orbital's quantum numbers to set initial conditions of integration
# spin can be +1 or -1
orb0 = hfnum.Orbital( 1, 1, 0, 0)
orb1 = hfnum.Orbital(-1, 1, 0, 0)
orb2 = hfnum.Orbital( 1, 2, 0, 0)

# now add it to the calculator
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)

# number of self-consistent iterations
NiterSCF = 20

# number of maximum iterations to loop over when scanning for the correct eigenenergy
Niter = 1000

# stop criteria on the energy
F0stop = 1e-5

# set velocity with which the self-consistent potentials are changed
# 0.7 works well, but other numbers can be tried in case of divergence
h.gammaSCF(0.7)

# actually solve the system
# you can set NiterSCF to 1 and call this many times to plot the orbitals in
# each step of the self-consistent iterations
h.solve(NiterSCF, Niter, F0stop)

# get list with r values for plotting later
r = np.asarray(h.getR())

# get orbitals shape
o = [np.asarray(orb0.get(0, 0)), np.asarray(orb1.get(0, 0)), np.asarray(orb2.get(0, 0))]

# get Coulomb attraction potential (just -Z/r)
v = h.getNucleusPotential()

# get direct and exchange potentials
vex = {}
vd = {}
vd[0] = h.getDirectPotential(0)
vex[0] = [h.getExchangePotential(0, 0), h.getExchangePotential(0, 1), h.getExchangePotential(0, 2)]
vd[1] = h.getDirectPotential(1)
vex[1] = [h.getExchangePotential(1, 0), h.getExchangePotential(1, 1), h.getExchangePotential(1, 2)]
vd[2] = h.getDirectPotential(2)
vex[2] = [h.getExchangePotential(2, 0), h.getExchangePotential(2, 1), h.getExchangePotential(2, 2)]

# one can now plot all the above as needed

# you can also save the result:
h.save("myresult.txt")

# and later load it again:
# to save the calculation state and continue, or plot the results later
h.load("myresult.txt")

```

# Installing packages necessary for compilation

```
sudo apt install libboost-dev* libboost-python*
```

# Compilation

```
cmake .
make
```

# Running

Try getting the energy and orbital shapes for Hydrogen with:

```
python share/test_H.py
```

For Helium, with:


```
python share/test_He.py
```

For Lithium, with:

```
python share/test_Li.py
```

For Beryllium, with:

```
python share/test_Be.py
```

