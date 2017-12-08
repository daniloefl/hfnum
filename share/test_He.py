
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 2
#dx = 1e-1/Z
#N = 170*Z
#rmin = 1e-5
#dx = 1e-2
#N = 1300
#rmin = 1e-4
dx = 1e-2
N = 2550
rmin = 1e-10
h = hfnum.HF(True, dx, int(N), rmin, Z)
h.addOrbital(0,  1, 1, 0, 0)
h.addOrbital(0, -1, 1, 0, 0)
h.method(2)

NiterSCF = 1
Niter = 100
F0stop = 1e-5
r = h.getR()
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 10):
  print "SCF it.", i
  h.gammaSCF(0.7)
  h.solve(NiterSCF, Niter, F0stop)

  r = np.asarray(h.getR())
  o = [np.asarray(h.getOrbital(0, 0, 0)), np.asarray(h.getOrbital(1, 0, 0))]
  v = h.getNucleusPotential()
  vex = {}
  vd = {}
  vd[0] = h.getDirectPotential(0)
  vex[0] = [h.getExchangePotential(0, 0), h.getExchangePotential(0, 1)]
  vd[1] = h.getDirectPotential(1)
  vex[1] = [h.getExchangePotential(1, 0), h.getExchangePotential(1, 1)]
  H1s = 2*np.exp(-r)

  m = next(i for i,v in enumerate(r) if v >= 5)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*o[1][:m], 'b:',  linewidth = 2, label = 'r*Orbital 1 (s)')
  plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[0][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[0][:m], 'b--', linewidth = 3, label = 'Direct (0)')
  plt.plot(r[:m], vex[0][0][:m], 'g:', linewidth = 3, label = 'Exchange (0,0)')
  plt.plot(r[:m], vex[0][1][:m], 'm:', linewidth = 3, label = 'Exchange (0,1)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[1][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[1][:m], 'b--', linewidth = 3, label = 'Direct (1)')
  plt.plot(r[:m], vex[1][0][:m], 'g:', linewidth = 3, label = 'Exchange (1,0)')
  plt.plot(r[:m], vex[1][1][:m], 'm:', linewidth = 3, label = 'Exchange (1,1)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()