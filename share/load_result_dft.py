
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

eV = 27.21138602

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

# read from this file
fname = "output/results_Li_dft.txt"

print "Loading result from file %s" % fname

# random initialisation
h = hfnum.DFT(fname)
#h.load(fname)

r = np.asarray(h.getR())
print "r:", r

for n in range(0, h.getNOrbitals()):
  print "Energy for orbital %10s (index=%2d, n=%2d, l=%2d, m=%2d, s=%2d): %10.6f Hartree = %15.8f eV" % (h.getOrbitalName(n), n, h.getOrbital_n(n), h.getOrbital_l(n), h.getOrbital_m(n), h.getOrbital_s(n), h.getOrbital_E(n), h.getOrbital_E(n)*eV)

E0 = h.getE0()
print "Total ground energy: %10.6f Hartree = %15.8f eV" % (E0, E0*eV)

o = []
v = h.getNucleusPotential()

m = next(i for i,v in enumerate(r) if v >= 10)

style = ['r-', 'b-', 'm-', 'c-', 'k-', 'g-']

n_up = h.getDensityUp()
n_dw = h.getDensityDown()
vhartree = h.getHartree()
vxup = h.getExchangeUp()
vxdw = h.getExchangeDown()

for n in range(0, h.getNOrbitals()):
  o.append(np.asarray(h.getCentral(n)))

# show orbitals
f = plt.figure()
i = 0
for idx in range(np.maximum(0, h.getNOrbitals()-len(style)), h.getNOrbitals()):
  plt.plot(r[:m], r[:m]*o[idx][:m], style[i], linewidth = 2, label = 'r*R[%6s]' % (h.getOrbitalName(idx)))
  i += 1
plt.legend()
plt.show()

f = plt.figure()
plt.plot(r[:m], r[:m]*n_up[:m], style[0], linewidth = 2, label = 'Electron density spin up')
plt.plot(r[:m], r[:m]*n_dw[:m], style[1], linewidth = 2, label = 'Electron density spin down')
plt.legend()
plt.show()

f = plt.figure()
i = 0
ymin = np.fabs(vhartree[0])
plt.plot(r[:m], v[:m], 'r-', linewidth = 2, label = 'Coulomb')
plt.plot(r[:m], vhartree[:m], 'b-', linewidth = 2, label = 'Hartree potential')
plt.plot(r[:m], vxup[:m], 'g-', linewidth = 2, label = 'Exchange up potential')
plt.plot(r[:m], vxdw[:m], 'm-', linewidth = 2, label = 'Exchange down potential')
plt.ylim((-ymin, ymin))
plt.legend()
plt.show()

