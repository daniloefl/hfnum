
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 9

# log grid
dx = 0.2e-1/2.0
N = 515*2
rmin = 1e-3
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb0 = hfnum.Orbital( 1, 0, "+-")
orb1 = hfnum.Orbital( 2, 0, "+-")
orb2 = hfnum.Orbital( 2, 1, "+-+-+N")
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)

NiterSCF = 40
Niter = 1000
F0stop = 1e-8
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.3)
h.solve(NiterSCF, Niter, F0stop)
h.save("output/results_F.txt")

