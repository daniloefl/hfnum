
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

Z = 1

# log grid
dx = 1e-2
N = 1900
rmin = 1e-7
h = hfnum.HF()
h.resetGrid(True, dx, int(N), rmin)
h.setZ(Z)
h.method(2)

orb = hfnum.Orbital(1, 1, 0, 0)

# adds p-like components
# that is, with this, the orbital 1s of Hydrogen will be modelled as:
# u(r, theta, phi) = u_s(r)*Y_00(theta, phi) + u_p1(r)*Y_1,0(theta, phi) + u_p2(r)*Y_1,-1(theta, phi) + u_p3(r)*Y_1,1(theta, phi)
# The system is solved to find the u_s, u_p1, u_p2, u_p3
# For Hydrogen 1s, we expect u_p1 = u_p2 = u_p3 = 0, but u_s != 0
# Still, this allows us to test the softwere to check that this basis expansion works in the energy finding algorithm
orb.addSphHarm(1,0)
orb.addSphHarm(1,-1)
orb.addSphHarm(1,1)

print "Adding orbital"
h.addOrbital(orb)

NiterSCF = 1
Niter = 100
F0stop = 1e-6
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0:5]
for i in range(0, 2):
  print "SCF it.", i
  h.gammaSCF(0.5)
  h.solve(NiterSCF, Niter, F0stop)

  o = [np.asarray(orb.get(0, 0))]
  o1 = [np.asarray(orb.get(1, 0)), np.asarray(orb.get(1, -1)), np.asarray(orb.get(1, 1))]
  v = h.getNucleusPotential()
  H1s = 2*np.exp(-r)

  m = next(i for i,v in enumerate(r) if v >= 10)
  f = plt.figure()
  plt.plot(r[:m], r[:m]*o[0][:m], 'r-',  linewidth = 2, label = 'r*Orbital 0 (s)')
  plt.plot(r[:m], r[:m]*o1[0][:m], 'b:',  linewidth = 2, label = 'r*Orbital 0 (p1)')
  plt.plot(r[:m], r[:m]*o1[1][:m], 'c:',  linewidth = 2, label = 'r*Orbital 0 (p2)')
  plt.plot(r[:m], r[:m]*o1[2][:m], 'm:',  linewidth = 2, label = 'r*Orbital 0 (p3)')
  plt.plot(r[:m], r[:m]*H1s[:m], 'g--', linewidth = 2, label = 'r*Hydrogen 1s')
  plt.legend()
  plt.show()

