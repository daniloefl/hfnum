
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 1

# log grid
dx = 1e-1
N = 120
rmin = 1e-4
h = hfnum.HF()
h.resetGrid(True, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb = hfnum.Orbital(1, 1, 0, 0)

h.addOrbital(orb)

NiterSCF = 1
Niter = 1
F0stop = 1e-6
r = np.asarray(h.getR())
print(r)
h.solve(NiterSCF, Niter, F0stop)

h.save('output/results_H_iss.txt')

o = [np.asarray(orb.getCentral())]
v = h.getNucleusPotential()
H1s = 2*np.exp(-r)

m = -1 #next(i for i,v in enumerate(r) if v >= 10)
f = plt.figure()
plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
plt.legend()
plt.show()

