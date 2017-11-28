
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

def savePot(r, V, name, forWF, actsOn):
  fout = open('pot_%s_%s_%s.dat' % (name, forWF, actsOn), 'w')
  fout.write("# name   %s\n" %(name))
  fout.write("# type   %s\n" %("exchange"))
  fout.write("# forWF  %s\n" %(forWF))
  fout.write("# actsOn %s\n" %(actsOn))
  for ib in range(0, len(r)):
    fout.write("%.16f     %.16f\n" % (r[ib], V[ib]))
  fout.close()

Z = 3

# log grid
#dx = 0.25e-3/Z
#N = 62000*Z
#rmin = 1e-5
dx = 0.5e-1
N = 470
rmin = 1e-9
for i in range(0, N):
  r = np.exp(np.log(rmin) + i*dx)
  V = -1.0/r
  a = 2*r*r*(-Z**2*0.5 - V) - 0.5**0.5
  if (a*dx)**2 > 6:
    print "Warning: (a*dx)^2 = ", (a*dx)**2, " > 6, at i = ", i, " for r = ", r, " --> can cause instabilities"
    break
h = hfnum.HF(True, dx, int(N), rmin, Z)

h.addOrbital(0,  1, 1, 0, 0)
h.addOrbital(0, -1, 1, 0, 0)
h.addOrbital(0,  1, 2, 0, 0)
#h.sparseMethod(False)

NiterSCF = 1
Niter = 1000
F0stop = 1e-10
r = h.getR()
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 20):
  print "SCF it.", i
  h.gammaSCF(0.4)
  h.solve(NiterSCF, Niter, F0stop)

  r = np.asarray(h.getR())
  o = [np.asarray(h.getOrbital(0, 0, 0)), np.asarray(h.getOrbital(1, 0, 0)), np.asarray(h.getOrbital(2, 0, 0))]
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

  #savePot(r, v, 'vnuc', '', '')

  #savePot(r, vd[0], 'vd', '1s1+', '1s1+')
  #savePot(r, vd[1], 'vd', '1s1-', '1s1-')
  #savePot(r, vd[2], 'vd', '2s1+', '2s1+')

  #savePot(r, vex[0][0], 'vxc', '1s1+', '1s1+')
  #savePot(r, vex[0][1], 'vxc', '1s1+', '1s1-')
  #savePot(r, vex[0][2], 'vxc', '1s1+', '2s1+')

  #savePot(r, vex[1][0], 'vxc', '1s1-', '1s1+')
  #savePot(r, vex[1][1], 'vxc', '1s1-', '1s1-')
  #savePot(r, vex[1][2], 'vxc', '1s1-', '2s1+')

  #savePot(r, vex[2][0], 'vxc', '2s1+', '1s1+')
  #savePot(r, vex[2][1], 'vxc', '2s1+', '1s1-')
  #savePot(r, vex[2][2], 'vxc', '2s1+', '2s1+')

  m = -1 #next(i for i,v in enumerate(r) if v >= 10)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*o[1][:m], 'b:',  linewidth = 2, label = 'r*Orbital 1 (s)')
  plt.plot(r[:m], r[:m]*o[2][:m], 'm:',  linewidth = 2, label = 'r*Orbital 2 (s)')
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
