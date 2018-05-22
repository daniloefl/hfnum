
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 1

# log grid
dx = 0.2e-1
N = 515
rmin = 1e-3
dx = 0.5e-1
N = 320
rmin = 1e-6
dx = 1.0/16.0
N = 120.0
rmin = np.exp(-4)/Z
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb = hfnum.Orbital(1, 0, "+N")

h.addOrbital(orb)

NiterSCF = 1
Niter = 100
F0stop = 1e-6
r = np.asarray(h.getR())
print r[0], r[-1]
h.solve(NiterSCF, Niter, F0stop)

h.save('output/results_H.txt')

o = [np.asarray(h.getCentral(0))]
v = h.getNucleusPotential()
H1s = 2*np.exp(-r)

m = next(i for i,v in enumerate(r) if v >= 10)
f = plt.figure()
plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
plt.legend()
plt.show()

