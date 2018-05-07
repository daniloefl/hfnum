
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 3

# log grid
dx = 0.5e-1/4
N = 422*4
rmin = 1e-8
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)

orb0 = hfnum.Orbital( 1, 0, "+-")
orb1 = hfnum.Orbital( 2, 0, "+ ")
h.addOrbital(orb0)
h.addOrbital(orb1)

h.method(3)

NiterSCF = 100
Niter = 200
F0stop = 1e-8
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.3)
h.solve(NiterSCF, Niter, F0stop)
h.save("output/results_Li.txt")

