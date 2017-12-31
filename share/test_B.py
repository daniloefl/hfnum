
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 5

# log grid
dx = 0.5*1e-1/Z
N = 2*225*Z
rmin = 1e-8
g = hfnum.Grid(True, dx, int(N), rmin)
h = hfnum.HF(g, Z)
h.method(2)

orb0 = hfnum.Orbital( 1, 1, 0, 0)
orb1 = hfnum.Orbital(-1, 1, 0, 0)
orb2 = hfnum.Orbital( 1, 2, 0, 0)
orb3 = hfnum.Orbital(-1, 2, 0, 0)
orb4 = hfnum.Orbital( 1, 2, 1, 0)
h.addOrbital(orb0)
h.addOrbital(orb1)
h.addOrbital(orb2)
h.addOrbital(orb3)
h.addOrbital(orb4)

NiterSCF = 40
Niter = 1000
F0stop = 1e-6
r = np.asarray(g.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
#for i in range(0, 20):
#  print "SCF it.", i
if True:
  h.gammaSCF(0.1)
  h.solve(NiterSCF, Niter, F0stop)

h.save("output/results_B.txt")

if True:
  o = [np.asarray(orb0.getCentral()), np.asarray(orb1.getCentral()), np.asarray(orb2.getCentral()), np.asarray(orb3.getCentral()), np.asarray(orb4.getCentral())]
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

