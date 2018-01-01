
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

# read from this file
fname = "output/results_C.txt"

# random initialisation
Z = 1
dx = 1e-1
N = 10
rmin = 1e-2
g = hfnum.Grid(True, dx, int(N), rmin)
h = hfnum.HF(g, Z)
h.method(2)
h.gammaSCF(0.1)

h.load(fname)

r = np.asarray(g.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]

o = []
v = h.getNucleusPotential()

m = next(i for i,v in enumerate(r) if v >= 10)

style = ['r-', 'b-', 'm-', 'c-', 'k-', 'g-']

vex = {}
vd = {}
for n in range(0, h.getNOrbitals()):
  o.append(np.asarray(h.getCentral(n)))

  vd[n] = h.getDirectPotential(n)

  tmp = []
  for n2 in range(0, h.getNOrbitals()):
    tmp.append(h.getExchangePotential(n, n2))
  vex[n] = tmp

# show orbitals
f = plt.figure()
i = 0
for n in range(h.getNOrbitals()-len(style), h.getNOrbitals()):
  plt.plot(r[:m], r[:m]*o[n][:m], style[i],  linewidth = 2, label = 'r*Orbital %d (l=%d, m=%d, s=%f)' % (n, h.getOrbital_n(n), h.getOrbital_l(n), h.getOrbital_m(n), h.getOrbital_s(n)))
  i += 1
plt.legend()
plt.show()

for n in range(0, h.getNOrbitals()):
  f = plt.figure()
  i = 0
  ymin = np.fabs(vd[0][0])
  plt.plot(r[:m], v[:m], 'r:', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[n][:m], 'b:', linewidth = 3, label = 'Direct (%d)' % n)
  for n2 in range(h.getNOrbitals()-len(style), h.getNOrbitals()):
    plt.plot(r[:m], vex[n][n2][:m], style[i], linewidth = 2, label = 'Exchange (%d,%d)' % (n, n2))
    i = i + 1
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

