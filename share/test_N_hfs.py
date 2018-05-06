
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 7

# log grid
dx = 0.5e-1/4
N = 421*4
rmin = 1e-8
h = hfnum.HFS()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb0 = hfnum.Orbital( 1, 0, 2)
orb1 = hfnum.Orbital( 2, 0, 2)
orb2 = hfnum.Orbital( 2, 1, 5)
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)

NiterSCF = 40
Niter = 1000
F0stop = 1e-6
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.1)
h.solve(NiterSCF, Niter, F0stop)
h.save("output/results_N_hfs.txt")

