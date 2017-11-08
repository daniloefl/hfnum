
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 2
dx = 1e-1/Z
N = 150*Z
rmin = 1e-5
h = hfnum.HF(dx, int(N), rmin, Z)
h.addOrbital(1, 1, 0, 0)
h.addOrbital(1, 1, 0, 0)
NiterSCF = 1
Niter = 100
F0stop = 1e-16
r = h.getR()
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 5):
  print "SCF it.", i
  h.gammaSCF(0.7)
  h.solve(NiterSCF, Niter, F0stop)

  r = np.asarray(h.getR())
  o1= np.asarray(h.getOrbital(0, 0, 0))
  o1l1m0  = np.asarray(h.getOrbital(0, 1, 0))
  o1l1mm1 = np.asarray(h.getOrbital(0, 1, -1))
  o1l1mp1 = np.asarray(h.getOrbital(0, 1,  1))
  o2 = np.asarray(h.getOrbital(1, 0, 0))
  o2l1m0  = np.asarray(h.getOrbital(1, 1, 0))
  o2l1mm1 = np.asarray(h.getOrbital(1, 1, -1))
  o2l1mp1 = np.asarray(h.getOrbital(1, 1,  1))
  v = h.getNucleusPotential()
  vd = h.getDirectPotential()
  H1s = 2*np.exp(-r)

  m = next(i for i,v in enumerate(r) if v >= 3)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o1[:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*o2[:m], 'b:',  linewidth = 2, label = 'r*Orbital 1 (s)')
  plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
  plt.legend()
  plt.show()

  f = plt.figure()
  plt.plot(r[:m], r[:m]*o1l1mm1[:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (p1)')
  plt.plot(r[:m], r[:m]*o1l1m0[:m], 'b-',  linewidth = 2, label = 'r*Orbital 0 (p2)')
  plt.plot(r[:m], r[:m]*o1l1mp1[:m], 'g-',  linewidth = 2, label = 'r*Orbital 0 (p3)')
  plt.plot(r[:m], r[:m]*o2l1mm1[:m], 'r:',  linewidth = 2, label = 'r*Orbital 1 (p1)')
  plt.plot(r[:m], r[:m]*o2l1m0[:m], 'b:',  linewidth = 2, label = 'r*Orbital 1 (p2)')
  plt.plot(r[:m], r[:m]*o2l1mp1[:m], 'g:',  linewidth = 2, label = 'r*Orbital 1 (p3)')
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 2, label = 'Coulomb')
  plt.plot(r[:m], vd[:m], 'b--', linewidth = 2, label = 'Direct')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()
