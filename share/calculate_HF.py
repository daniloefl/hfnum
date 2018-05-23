
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

import re

Z = 1

print("Please enter the atomic number.")

Z = raw_input("> ")
Z = float(Z)


# log grid
dx = 1.0/16.0*0.25
N = 130.0*4
rmin = np.exp(-4)/Z
h = hfnum.HF()
h.resetGrid(1, dx, int(N), rmin)
h.setZ(Z)
h.method(3)

print("Please enter the electron configuration in the format: 1s2 2s2 2p3")
config = raw_input("> ")
config = config.split()
r = re.compile("([0-9]+)([spdfghSPDFGH]+)([0-9]+)")
orb = []
for i in config:
  m = r.match(i)
  n = int(m.group(1))
  l_str = m.group(2)
  l_str = l_str.lower()
  l = 0
  if l_str == 's': l = 0
  elif l_str == 'p': l = 1
  elif l_str == 'd': l = 2
  elif l_str == 'f': l = 3
  elif l_str == 'g': l = 4
  elif l_str == 'h': l = 5
  c = ['n']*(2*(2*l+1))
  mult = int(m.group(3))
  for k in range(0, min([(2*l+1), mult])):
    pos = 2*k
    c[pos] = '+'
  for k in range(0, min([(2*l+1), mult - 2*l - 1])):
    pos = 1 + 2*k
    c[pos] = '-'
  c = "".join(c)
  orb.append(hfnum.Orbital(n, l, c))
  print("Added orbital with (n, l) = (%d, %d) and electron configuration %s" % (n, l, c))
  h.addOrbital(orb[-1])


print("Please enter the output file name.")
foutname = raw_input("> ")

NiterSCF = 40
Niter = 100
F0stop = 1e-6
r = np.asarray(h.getR())
print "Last r:", r[-1]
print "First r:", r[0]
h.gammaSCF(0.3)
h.solve(NiterSCF, Niter, F0stop)

h.save(foutname)

