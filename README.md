# hfnum

Hartree-Fock calculation in C++ using a numerical Grid. Based on hfpython repository.
It currently can use linear or logarithmic Grids, but only logarithmic Grids have been observed to work with acceptable precision.
It can also either assume initial conditions and perform the equation integration , or it can build an NxN matrix and solve a sparse
matrix system using the matrix Numerov method.

Although the iterative method, which assumes initial conditions works well for the Hydrogen and Helium, it fails in Lithium, where an exchange potential
appears for the first time. The sparse matrix Numerov method is slower but works in Lithium, since it does not implement initial conditions assumptions.

A solution must be found for the iterative method, so that it can make a smarter guess for the initial conditions.

The software is a Python library, where the calculations are done in C++, but the configuration of the parameters is done in Python.
Example Python configurations for the Hydrogen, Helium, Lithium and Beryllium can be seen in the share directory.
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

# atomic number:
Z = 3

# log grid
# r = exp(log(rmin) + dx*i), where  i = 0..N-1
# change the Grid parameters below
dx = 0.5e-1       # Grid step
N = 390           # number of points
rmin = 1e-7       # first point in the Grid

# initialise library with the Grid parameters
# the first parameter tells it whether one should use the logarithmic Grid
# the linear Grid works poorly, so it is recommended to keep this always in True
h = hfnum.HF(True, dx, int(N), rmin, Z)

# call addOrbital as many times as needed to
# add an orbital
# the syntax is the following:
# h.addOrbital(Lmax, spin, n l, m)
# n, l and m are the orbital's quantum numbers to set initial conditions of integration
# spin can be +1 or -1
# Lmax indicates how many spherical harmonics are used in this orbital's representation
h.addOrbital(0,  1, 1, 0, 0)
h.addOrbital(0, -1, 1, 0, 0)
h.addOrbital(0,  1, 2, 0, 0)

# number of self-consistent iterations
NiterSCF = 20

# number of maximum iterations to loop over when scanning for the correct eigenenergy
Niter = 1000

# stop criteria on minimization quantity (1e-10 is recommended
F0stop = 1e-10

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
o = [np.asarray(h.getOrbital(0, 0, 0)), np.asarray(h.getOrbital(1, 0, 0)), np.asarray(h.getOrbital(2, 0, 0))]

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
```

To switch from the sparse matrix Numerov method and the iterative method, one can use this configuration option
before calling `solve`:

```
h.sparseMethod(False)     #  to de-activate the sparse matrix Numerov method
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

