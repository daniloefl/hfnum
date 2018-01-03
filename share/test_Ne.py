
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 10

# log grid
dx = 1e-1/Z
N = 225*Z
rmin = 1e-8
h = hfnum.HF()
h.resetGrid(True, dx, int(N), rmin)
h.setZ(Z)
h.method(2)

orb0 = hfnum.Orbital( 1, 1, 0, 0)
orb1 = hfnum.Orbital(-1, 1, 0, 0)
orb2 = hfnum.Orbital( 1, 2, 0, 0)
orb3 = hfnum.Orbital(-1, 2, 0, 0)
orb4 = hfnum.Orbital( 1, 2, 1, 0)
orb5 = hfnum.Orbital(-1, 2, 1, 0)
orb6 = hfnum.Orbital( 1, 2, 1, -1)
orb7 = hfnum.Orbital(-1, 2, 1, -1)
orb8 = hfnum.Orbital( 1, 2, 1,  1)
orb9 = hfnum.Orbital(-1, 2, 1,  1)
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)
h.addOrbital(orb3)
h.addOrbital(orb4)
h.addOrbital(orb5)
h.addOrbital(orb6)
h.addOrbital(orb7)
h.addOrbital(orb8)
h.addOrbital(orb9)

NiterSCF = 40
Niter = 1000
F0stop = 1e-6
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.1)
h.solve(NiterSCF, Niter, F0stop)
h.save("output/results_Ne.txt")

