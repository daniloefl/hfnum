
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 2
dx = 0.5e-1/4
N = 421*4
rmin = 1e-8
h = hfnum.HFS()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
orb0 = hfnum.Orbital( 1, 0, "+-")
h.method(3)
h.addOrbital(orb0)

NiterSCF = 40
Niter = 100
F0stop = 1e-8
r = h.getR()
r = np.asarray(r)
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.1)
h.solve(NiterSCF, Niter, F0stop)
h.save('output/results_He_hfs.txt')

