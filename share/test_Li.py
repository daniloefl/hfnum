
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 3

# log grid
dx = 0.5e-1
N = 421
rmin = 1e-8
h = hfnum.HF()
h.resetGrid(True, dx, int(N), rmin)
h.setZ(Z)

orb0 = hfnum.Orbital( 1, 1, 0, 0)
orb1 = hfnum.Orbital(-1, 1, 0, 0)
orb2 = hfnum.Orbital( 1, 2, 0, 0)
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)

h.method(2)

NiterSCF = 40
Niter = 200
F0stop = 1e-5
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.4)
h.solve(NiterSCF, Niter, F0stop)
h.save("output/results_Li.txt")

