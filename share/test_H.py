
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 1

# log grid
#dx = 1e-1/Z
#N = 150*Z
#rmin = 1e-4
dx = 1e-2
N = 2550
rmin = 1e-10
h = hfnum.HF(True, dx, int(N), rmin, Z)
h.sparseMethod(False)

# linear grid
#dx = 1e-4/Z
#N = 50000*Z
#rmin = 1e-4
#h = hfnum.HF(False, dx, int(N), rmin, Z)

h.addOrbital(0,  1, 1, 0, 0)

NiterSCF = 1
Niter = 100
F0stop = 1e-5
#F0stop = 1e-6
r = h.getR()
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 2):
  print "SCF it.", i
  h.gammaSCF(0.5)
  h.solve(NiterSCF, Niter, F0stop)

  r = np.asarray(h.getR())
  o = [np.asarray(h.getOrbital(0, 0, 0))]
  v = h.getNucleusPotential()
  H1s = 2*np.exp(-r)

  m = next(i for i,v in enumerate(r) if v >= 5)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
  plt.legend()
  plt.show()

