
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 5

# log grid
dx = 1e-1/Z
N = 300*Z
rmin = 1e-12
for i in range(0, N):
  r = np.exp(np.log(rmin) + i*dx)
  V = -1.0/r
  a = 2*r*r*(-Z**2*0.5 - V) - 0.5**0.5
  if (a*dx)**2 > 6:
    print "Warning: (a*dx)^2 = ", (a*dx)**2, " > 6, at i = ", i, " for r = ", r, " --> can cause instabilities"
    break
h = hfnum.HF(True, dx, int(N), rmin, Z)
h.sparseMethod(False)

h.addOrbital(0,  1, 1, 0, 0)
h.addOrbital(0, -1, 1, 0, 0)
h.addOrbital(0,  1, 2, 0, 0)
h.addOrbital(0, -1, 2, 0, 0)
h.addOrbital(1,  1, 2, 1, 0)

NiterSCF = 1
Niter = 1000
F0stop = 1e-5
r = h.getR()
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 20):
  print "SCF it.", i
  h.gammaSCF(0.7)
  h.solve(NiterSCF, Niter, F0stop)

  r = np.asarray(h.getR())
  o = [np.asarray(h.getOrbital(0, 0, 0)), np.asarray(h.getOrbital(1, 0, 0)), np.asarray(h.getOrbital(2, 0, 0)), np.asarray(h.getOrbital(3, 0, 0)), np.asarray(h.getOrbital(4, 0, 0))]
  v = h.getNucleusPotential()
  vex = {}
  vd = {}
  vd[0] = h.getDirectPotential(0)
  vex[0] = [h.getExchangePotential(0, 0), h.getExchangePotential(0, 1), h.getExchangePotential(0, 2), h.getExchangePotential(0, 3), h.getExchangePotential(0, 4)]
  vd[1] = h.getDirectPotential(1)
  vex[1] = [h.getExchangePotential(1, 0), h.getExchangePotential(1, 1), h.getExchangePotential(1, 2), h.getExchangePotential(1, 3), h.getExchangePotential(1, 4)]
  vd[2] = h.getDirectPotential(2)
  vex[2] = [h.getExchangePotential(2, 0), h.getExchangePotential(2, 1), h.getExchangePotential(2, 2), h.getExchangePotential(2, 3), h.getExchangePotential(2, 4)]
  vd[3] = h.getDirectPotential(3)
  vex[3] = [h.getExchangePotential(3, 0), h.getExchangePotential(3, 1), h.getExchangePotential(3, 2), h.getExchangePotential(3, 3), h.getExchangePotential(3, 4)]
  vd[4] = h.getDirectPotential(4)
  vex[4] = [h.getExchangePotential(4, 0), h.getExchangePotential(4, 1), h.getExchangePotential(4, 2), h.getExchangePotential(4, 3), h.getExchangePotential(4, 4)]
  H1s = 2*np.exp(-r)

  m = next(i for i,v in enumerate(r) if v >= 10)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*o[1][:m], 'b:',  linewidth = 2, label = 'r*Orbital 1 (s)')
  plt.plot(r[:m], r[:m]*o[2][:m], 'm:',  linewidth = 2, label = 'r*Orbital 2 (s)')
  plt.plot(r[:m], r[:m]*o[3][:m], 'c:',  linewidth = 2, label = 'r*Orbital 3 (s)')
  plt.plot(r[:m], r[:m]*o[4][:m], 'k:',  linewidth = 2, label = 'r*Orbital 4 (p)')
  plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[0][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[0][:m], 'b-', linewidth = 3, label = 'Direct (0)')
  plt.plot(r[:m], vex[0][0][:m], 'g--', linewidth = 3, label = 'Exchange (0,0)')
  plt.plot(r[:m], vex[0][1][:m], 'm--', linewidth = 3, label = 'Exchange (0,1)')
  plt.plot(r[:m], vex[0][2][:m], 'c--', linewidth = 3, label = 'Exchange (0,2)')
  plt.plot(r[:m], vex[0][3][:m], 'k--', linewidth = 3, label = 'Exchange (0,3)')
  plt.plot(r[:m], vex[0][4][:m], 'y--', linewidth = 3, label = 'Exchange (0,4)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[1][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[1][:m], 'b-', linewidth = 3, label = 'Direct (1)')
  plt.plot(r[:m], vex[1][0][:m], 'g--', linewidth = 3, label = 'Exchange (1,0)')
  plt.plot(r[:m], vex[1][1][:m], 'm--', linewidth = 3, label = 'Exchange (1,1)')
  plt.plot(r[:m], vex[1][2][:m], 'c--', linewidth = 3, label = 'Exchange (1,2)')
  plt.plot(r[:m], vex[1][3][:m], 'k--', linewidth = 3, label = 'Exchange (1,3)')
  plt.plot(r[:m], vex[1][4][:m], 'y--', linewidth = 3, label = 'Exchange (1,4)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[2][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[2][:m], 'b-', linewidth = 3, label = 'Direct (2)')
  plt.plot(r[:m], vex[2][0][:m], 'g--', linewidth = 3, label = 'Exchange (2,0)')
  plt.plot(r[:m], vex[2][1][:m], 'm--', linewidth = 3, label = 'Exchange (2,1)')
  plt.plot(r[:m], vex[2][2][:m], 'c--', linewidth = 3, label = 'Exchange (2,2)')
  plt.plot(r[:m], vex[2][3][:m], 'k--', linewidth = 3, label = 'Exchange (2,3)')
  plt.plot(r[:m], vex[2][4][:m], 'y--', linewidth = 3, label = 'Exchange (2,4)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[3][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[3][:m], 'b-', linewidth = 3, label = 'Direct (3)')
  plt.plot(r[:m], vex[3][0][:m], 'g--', linewidth = 3, label = 'Exchange (3,0)')
  plt.plot(r[:m], vex[3][1][:m], 'm--', linewidth = 3, label = 'Exchange (3,1)')
  plt.plot(r[:m], vex[3][2][:m], 'c--', linewidth = 3, label = 'Exchange (3,2)')
  plt.plot(r[:m], vex[3][3][:m], 'k--', linewidth = 3, label = 'Exchange (3,3)')
  plt.plot(r[:m], vex[3][4][:m], 'y--', linewidth = 3, label = 'Exchange (3,4)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[4][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[4][:m], 'b-', linewidth = 3, label = 'Direct (4)')
  plt.plot(r[:m], vex[4][0][:m], 'g--', linewidth = 3, label = 'Exchange (4,0)')
  plt.plot(r[:m], vex[4][1][:m], 'm--', linewidth = 3, label = 'Exchange (4,1)')
  plt.plot(r[:m], vex[4][2][:m], 'c--', linewidth = 3, label = 'Exchange (4,2)')
  plt.plot(r[:m], vex[4][3][:m], 'k--', linewidth = 3, label = 'Exchange (4,3)')
  plt.plot(r[:m], vex[4][4][:m], 'y--', linewidth = 3, label = 'Exchange (4,4)')
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()
