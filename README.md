# hfnum

Hartree-Fock calculation in C++ using a numerical Grid. Based on hfpython repository.
It currently can use linear or logarithmic Grids, but only logarithmic Grids have been observed to work with acceptable precision.

Four methods are available to solve the differential equation:
  * method 0: Sparse Numerov Matrix method
    * Creates one numerical equation per differential equation and Grid point and puts them all in an NxN sparse matrix. Extra equations are created to force the normalisation of the eigenfunctions to be 1. Since the normalisation condition is non-linear, the system is resolved using the Newton-Raphson method, by calculating the Jacobian matrix of partial derivatives and changing the energy and function values according to -X inverse(Jacobian), where X is the column-vector of wavefunction values and energies. This method is extremely slow, but it is simple and assumes only that the wave function first and last values are zero.
  * method 1: Iterative Numerov Method using Gordon's method to guess initial conditions
    * The system can be solved (up to a normalisation) using Numerov equation to get the third point based on the two points before it. However, we need two initial conditions and choosing the wrong initial conditions (particularly in non-symmetric systems, such as atoms with more than 2 electrons) can lead to divergence. Gordon's solution tries a set of linearly independent solutions and uses a clever method to discover the correct initial conditions. It is described here: http://aip.scitation.org/doi/pdf/10.1063/1.436421
  * method 2: Iterative renormalised method (stable)
    * This method is an extension of the method proposed by Gordon. The method is only re-written in a different way using the ratio of solutions normalised by the differential equation coefficients. This procedure avoids overflows, which happen in method 1. It is recommended and it is explained here: http://aip.scitation.org/doi/pdf/10.1063/1.436421
  * method 3: Standard Numerov method with non-homogeneus term (faster, default)
    * This method solves the equations using the Numerov method multiplying out the terms that depend on other orbitals and leaving them as an independent non-homogeneous term. This procedure is repeated several times to achieve consistency before recalculating the energy and moving to the potential self-consistency step. This method is a simple and fast extension of the Numerov standard method, but it does not often converge easily. One paper using this method worth reading is: https://www.sciencedirect.com/science/article/pii/0010465576900400

The software is a Python library, where the calculations are done in C++, but the configuration of the parameters is done in Python.
Example Python configurations for the Hydrogen, Helium, Lithium, Beryllium, Boron and Carbon can be seen in the share directory.
Note that the central potential approximation is used to solve the equations in the radial variable, so the energies found will only be a first approximation.
Perturbative corrections can be applied further using the code in src/NonCentralCorrection.cxx (but it has some bugs: TODO).

The actual equation set up can be done either using several methods, which implement different potential models:
  * Hartree-Fock with central potential approximation: this is implemented in the hfnum.HF class. It projects the potentials in the spherical harmonic of the orbitla being calculated in case of non-filled shells. This should give the most accurate result, but it is the slowest. The examples below show this method being used.
  * Hartree-Fock-Slater method: this method is described in "A Simplification of the Hartree-Fock Method", by J. C. Slater ( https://journals.aps.org/pr/abstract/10.1103/PhysRev.81.385 ). It uses the free electron gas approximation to estimate the exchange potential, eliminating the non-homogeneous terms in the equations. This is implemented in the hfnum.HFS class. This gives results that may be very off for low Z atoms (ie: He), due to this approximation, but the equation is easily solved with it. For this reason, it is worth using this method and applying the NonCentralCorrection with share/load_and_correct_result.py to get a reasonable solution.
  * Density Functional Theory method: this is a simple implementation of DFT, splitting the system in spin up and spin down electrons and estimating the potentials using the charge density for spin up and spin down electrons. The exchange potential used is calculating using the Local Density Approximation. This method is quite fast and it is implemented in the hfnum.DFT class.

The examples below can all be done using any of the Hartree-Fock, Hartree-Fock-Slater or Density Functional Theory methods, by simply replacing the
class constructor from hfnum.HF to hfnum.HFS or hfnum.DFT, as appropriate.


# Installing packages necessary for compilation

```
sudo apt install libboost-dev* libboost-python*
```

# Compilation

```
cmake .
make
```

# How to run it

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
dx = 0.5e-1/4       # Grid step
N = 421*4           # number of points
rmin = 1e-8         # first point in the Grid

# this is the main solver
# change this to hfnum.HFS for the Hartree-Fock-Slater method
# or hfnum.DFT for the Density Functional Theory approach
h = hfnum.HF()

# initialise library with the Grid parameters
# the first parameter tells it whether one should use the logarithmic Grid
# the linear Grid works poorly, so it is recommended to keep this always in 1
h.resetGrid(1, dx, int(N), rmin)

# set atomic number
h.setZ(Z)

# use this (default) method for speed, but change it to 2 for more stability
h.method(3)

# create an Orbital as many times as needed
# the syntax is the following:
# myVar = hfnum.Orbital(n, l, electronDistribution)
# n, l are the orbital's quantum numbers to set initial conditions of integration
# electron distribution is a string with 2*(2*l + 1) characters, which must each be one of "+", "-" or "N".
# It specifies the filled m_s and m_l shells in the order (m_l = -l, m_s = +1), (m_l = -l, m_s = -1), (m_l = -l+1, m_s = +1), etc.
# For example: "+-+NNN" specifies that there are 3 electrons up and one down. The electron down is in m_l = -1.
# The three electrons up are in m_l = -1, 0 and 1.
orb0 = hfnum.Orbital( 1, 0, "+-")
orb1 = hfnum.Orbital( 2, 0, "+N")

# now add it to the calculator
h.addOrbital(orb0)
h.addOrbital(orb1)

# Note: all electrons in an Orbital have the same radial dependence
# If you want two electrons to have independent radial wave functions above, you can simply split them in two Orbital objects:
# 
# orb0 = hfnum.Orbital( 1, 0, "+N")
# orb1 = hfnum.Orbital( 1, 0, "N-")
# orb2 = hfnum.Orbital( 2, 0, "+N")
# h.addOrbital(orb0)
# h.addOrbital(orb1)
# h.addOrbital(orb2)
#
# This allows their radial functions and energies to vary independently.
# It will take longer, but might give more freedom to achieve a better approximation.

# number of self-consistent iterations
NiterSCF = 20

# number of maximum iterations to loop over when scanning for the correct eigenenergy
Niter = 100

# stop criteria on the energy
F0stop = 1e-6

# set velocity with which the self-consistent potentials are changed
# 0.1 works well, but other numbers can be tried in case of divergence
h.gammaSCF(0.1)

# actually solve the system
# you can set NiterSCF to 1 and call this many times to plot the orbitals in
# each step of the self-consistent iterations
h.solve(NiterSCF, Niter, F0stop)

# get list with r values for plotting later
r = np.asarray(h.getR())

# get orbitals shape
o = [np.asarray(orb0.get()), np.asarray(orb1.get())]

# get Coulomb attraction potential (just -Z/r)
v = h.getNucleusPotential()

# get direct and exchange potentials
vex = {}
vd = {}
vd[0] = h.getDirectPotential(0)
vex[0] = [h.getExchangePotential(0, 0), h.getExchangePotential(0, 1)]
vd[1] = h.getDirectPotential(1)
vex[1] = [h.getExchangePotential(1, 0), h.getExchangePotential(1, 1)]

# one can now plot all the above as needed

# you can also save the result:
h.save("myresult.txt")

# and later load it again:
# to save the calculation state and continue, or plot the results later
h.load("myresult.txt")

```

After the results have been produced (see examples in the `output` directory), one can also read the HF results and
estimate corrections to the assumptions made using perturbation theory. The following shows how the non-central correction
can be calculated from a previously saved Lithium atom calculation. In general, one can use `share/load_and_correct_result.py` for this.

```
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import hfnum

# read from this file
fname = "output/results_Li.txt"
 
h = hfnum.NonCentralCorrection()
h.load(fname)
h.correct()
pert_E = h.getCorrectedE()

eV = 27.21138602

for n in range(0, h.getNOrbitals()):
  print "Energy for orbital %10s: %10.6f Hartree = %15.8f eV, first order non-central correction: %10.6f Hartree = %15.8f eV" % (h.getOrbitalName(n), h.getOrbital_E(n), h.getOrbital_E(n)*eV, pert_E[n], pert_E[n]*eV)

```

The results can then be read and corrected using the hfnum.NonCentralCorrection as follows (an example can be found in share/load_and_correct_result.py):

```
h = hfnum.NonCentralCorrection()
h.load("output/result_He_hfs.txt")
h.correct()

# get corrected eigenvalues
pert_E = h.getCorrectedE()
# get coefficients to apply to uncorrected orbitals to get the corrected ones
coeff = h.getCoefficients()

# get the corrected ground state energy
E0 = h.getE0()

# get the uncorrected ground state energy
E0uncorr = h.getE0Uncorrected()
```

# Examples

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

# Reading saved results

Existing results in the `output` directory can be read using:

```
python share/load_result.py
```

In addition, the non-central potential correction can be calculated using perturbation theory with:

```
python share/load_and_correct_result.py
```

