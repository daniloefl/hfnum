
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 2
dx = 1.0/16.0*0.5
N = 120.0*2
rmin = np.exp(-4)/Z
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
orb0 = hfnum.Orbital( 1, 0, "+-")
h.method(4)
h.addOrbital(orb0)

NiterSCF = 200
Niter = 100
F0stop = 1e-8
r = h.getR()
r = np.asarray(r)
print "Last r:", r[-1]
print "First r:", r[0:5]
h.gammaSCF(0.7)
h.solve(NiterSCF, Niter, F0stop)
h.save('output/results_He.txt')

