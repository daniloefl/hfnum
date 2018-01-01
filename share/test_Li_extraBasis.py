
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 3

# log grid
dx = 1e-1/Z
N = 170*Z
rmin = 1e-6
h = hfnum.HF()
h.resetGrid(True, dx, int(N), rmin)
h.setZ(Z)

orb0 = hfnum.Orbital( 1, 1, 0, 0)
orb1 = hfnum.Orbital(-1, 1, 0, 0)
orb2 = hfnum.Orbital( 1, 2, 0, 0)
orb2.addSphHarm(1,  0)
#orb2.addSphHarm(1, -1)
#orb2.addSphHarm(1,  1)
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)

h.method(2)

NiterSCF = 1
Niter = 200
F0stop = 1e-5
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 20):
  print "SCF it.", i
  h.gammaSCF(0.4)
  h.solve(NiterSCF, Niter, F0stop)

  o = [np.asarray(orb0.get(0, 0)), np.asarray(orb1.get(0, 0)), np.asarray(orb2.get(0, 0))]
  #o2p = [np.asarray(orb2.get(1, -1)), np.asarray(orb2.get(1, 0)), np.asarray(orb2.get(1, 1))]
  o2p = [np.asarray(orb2.get(1, 0))]
  v = h.getNucleusPotential()
  vex = {}
  vd = {}
  vd[0] = h.getDirectPotential(0)
  vex[0] = [h.getExchangePotential(0, 0), h.getExchangePotential(0, 1), h.getExchangePotential(0, 2)]
  vd[1] = h.getDirectPotential(1)
  vex[1] = [h.getExchangePotential(1, 0), h.getExchangePotential(1, 1), h.getExchangePotential(1, 2)]
  vd[2] = h.getDirectPotential(2)
  vex[2] = [h.getExchangePotential(2, 0), h.getExchangePotential(2, 1), h.getExchangePotential(2, 2)]
  H1s = 2*np.exp(-r)

  m = next(i for i,v in enumerate(r) if v >= 10)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*o[1][:m], 'g-',  linewidth = 2, label = 'r*Orbital 1 (s)')
  plt.plot(r[:m], r[:m]*o[2][:m], 'b-',  linewidth = 2, label = 'r*Orbital 2 (s)')
  plt.plot(r[:m], r[:m]*o2p[0][:m], 'r--',  linewidth = 1, label = 'r*Orbital 2 (p1)')
  #plt.plot(r[:m], r[:m]*o2p[1][:m], 'g--',  linewidth = 2, label = 'r*Orbital 2 (p2)')
  #plt.plot(r[:m], r[:m]*o2p[2][:m], 'b--',  linewidth = 3, label = 'r*Orbital 2 (p3)')
  #plt.plot(r[:m], r[:m]*H1s[:m], 'gv', linewidth = 2, label = 'r*Hydrogen 1s')
  plt.legend()
  plt.show()

  f = plt.figure()
  ymin = np.fabs(vd[0][0])
  plt.plot(r[:m], v[:m], 'r-', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[0][:m], 'b-', linewidth = 3, label = 'Direct (0)')
  plt.plot(r[:m], vex[0][0][:m], 'g--', linewidth = 3, label = 'Exchange (0,0)')
  plt.plot(r[:m], vex[0][1][:m], 'm--', linewidth = 3, label = 'Exchange (0,1)')
  plt.plot(r[:m], vex[0][2][:m], 'c--', linewidth = 3, label = 'Exchange (0,2)')
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
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()
