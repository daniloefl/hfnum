
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
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

orb = hfnum.Orbital(2, 1, "+NNNNN")

h.addOrbital(orb)

NiterSCF = 1
Niter = 100
F0stop = 1e-6
r = np.asarray(h.getR())
h.solve(NiterSCF, Niter, F0stop)

h.save('output/results_H_2p.txt')

o = [np.asarray(orb.getCentral())]
v = h.getNucleusPotential()
H1s = 2*np.exp(-r)
H2s = 1.0/(2.0*np.sqrt(2))*(2 - r)*np.exp(-r/2.0)
H2p = 1.0/(2.0*np.sqrt(6))*r*np.exp(-r/2.0)

m = -1 #next(i for i,v in enumerate(r) if v >= 10)
f = plt.figure()
plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
plt.plot(r[:m], r[:m]*H2s[:m], 'b--', linewidth = 2, label = 'r*Hydrogen 2s')
plt.plot(r[:m], r[:m]*H2p[:m], 'm--', linewidth = 2, label = 'r*Hydrogen 2p')
plt.legend()
plt.show()

