
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 1

h = hfnum.RHF()
h.setZ(Z)
h.addOrbital(1, 0, 0,  1)
#h.addOrbital(1, 0, 0, -1)
h.Nscf(10)
h.solve()

r = []
for i in range(0, 130):
  r.append(np.exp(np.log(1e-4) + i*1e-1))


o_up = []
for i in range(0, h.getNOrbitals(1)):
  l = h.getOrbital_l(int(i), int(1))
  m = h.getOrbital_m(i, 1)
  o_up.append(h.getOrbital(i, 1, l, m, r))
  o_up[-1] = np.asarray(o_up[-1])

o_dw = []
for i in range(0, h.getNOrbitals(-1)):
  l = h.getOrbital_l(i, -1)
  m = h.getOrbital_m(i, -1)
  o_dw.append(h.getOrbital(i, -1, l, m, r))
  o_dw[-1] = np.asarray(o_dw[-1])

r = np.asarray(r)

m = next(i for i,v in enumerate(r) if v >= 5)
f = plt.figure()
plt.plot(r[:m], 2*np.exp(-np.asarray(r[:m])), 'b:',  linewidth = 2, label = 'H 1s')
plt.plot(r[:m], o_up[0][:m], 'r--',  linewidth = 2, label = 'Orbital 0 up (s)')
#plt.plot(r[:m], o_dw[0][:m], 'g--',  linewidth = 2, label = 'Orbital 0 dw (s)')
plt.legend()
plt.show()

