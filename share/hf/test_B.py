
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 5

# log grid
dx = 1.0/16.0*0.5
N = 120.0*2
rmin = np.exp(-4)/Z
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb0 = hfnum.Orbital( 1, 0, "+-")
orb1 = hfnum.Orbital( 2, 0, "+-")
orb2 = hfnum.Orbital( 2, 1, "+NNNNN")
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)

NiterSCF = 40
Niter = 100
F0stop = 1e-8
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.3)
h.solve(NiterSCF, Niter, F0stop)
h.save("output/results_B.txt")

