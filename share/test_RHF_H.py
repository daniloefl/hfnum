
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
h.solve()

r = []
for i in range(0, 150):
  r.append(np.exp(np.log(1e-4) + i*1e-1))

o_up = []
o_up.append(h.getOrbital(0, 1, 0, 0, r))

m = next(i for i,v in enumerate(r) if v >= 10)
f = plt.figure()
plt.plot(r[:m], 2*np.exp(-np.asarray(r[:m])), 'b:',  linewidth = 2, label = 'H 1s')
plt.plot(r[:m], o_up[0][:m], 'r--',  linewidth = 2, label = 'Orbital 0 (s)')
plt.legend()
plt.show()

