
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 1

# log grid
dx = 0.5e-1
N = 421
rmin = 1e-8
dx = 0.1e-1
N = 1300
rmin = 1e-4
dx = 1.0/16.0/Z
N = 120*Z
rmin = np.exp(-4)/Z
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb = hfnum.Orbital(1, 1, 0, 0)

h.addOrbital(orb)

NiterSCF = 1
Niter = 100
F0stop = 1e-6
r = np.asarray(h.getR())
print r[0], r[-1]
h.solve(NiterSCF, Niter, F0stop)

h.save('output/results_H.txt')

o = [np.asarray(orb.getCentral())]
v = h.getNucleusPotential()
H1s = 2*np.exp(-r)

m = next(i for i,v in enumerate(r) if v >= 10)
f = plt.figure()
plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
plt.legend()
plt.show()

