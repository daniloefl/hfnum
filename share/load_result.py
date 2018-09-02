
import sys
sys.path.append("../lib/")
sys.path.append("lib/")

import numpy as np
import hfnum

import seaborn
import matplotlib.pyplot as plt

# read from this file
fname = "output/results_He.txt"
 
import readline, glob
def complete(text, state):
  return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)
print('What is the input file to load?')
print('Examples are any result*.txt file in the output directory. The ones with dft in their name are to be used with load_result_dft.py instead, so please do not load them with this script.')
print('[feel free to use TAB to auto-complete]')
fname = raw_input('')

#print "Loading result from file %s" % fname

# random initialisation
h = hfnum.HF(fname)
#h.load(fname)

r = np.asarray(h.getR())
#print "r:", r

for n in range(0, h.getNOrbitals()):
  print("Energy for orbital %10s: %10.6f Hartree = %15.8f eV" % (h.getOrbitalName(n), h.getOrbital_E(n), h.getOrbital_E(n)*hfnum.eV))

E0 = h.getE0()
print("Total ground energy: %10.6f Hartree = %15.8f eV" % (E0, E0*hfnum.eV))

o = []
v = h.getNucleusPotential()

m = -1 #next(i for i,v in enumerate(r) if v >= 10)

style = ['r-', 'b-', 'm-', 'c-', 'k-', 'g-']

vex = {}
vd = {}
for n in range(0, h.getNOrbitals()):
  o.append(np.asarray(h.getCentral(n)))

  vd[n] = h.getDirectPotential(n)

  tmp = []
  for n2 in range(0, h.getNOrbitals()):
    tmp.append(h.getExchangePotential(n, n2))
  vex[n] = tmp

# show orbitals
f = plt.figure()
i = 0
for n in range(np.maximum(0, h.getNOrbitals()-len(style)), h.getNOrbitals()):
  plt.plot(r[:m], r[:m]*o[n][:m], style[i], linewidth = 2, label = 'r*R[%6s]' % (h.getOrbitalName(n)))
  i += 1
plt.legend()
plt.show()

for n in range(0, h.getNOrbitals()):
  f = plt.figure()
  i = 0
  ymin = np.fabs(vd[0][0])
  plt.plot(r[:m], v[:m], 'r:', linewidth = 3, label = 'Coulomb')
  plt.plot(r[:m], vd[n][:m], 'b:', linewidth = 3, label = 'Direct [%s]' % h.getOrbitalName(n))
  for n2 in range(np.maximum(0, h.getNOrbitals()-len(style)), h.getNOrbitals()):
    plt.plot(r[:m], vex[n][n2][:m], style[i], linewidth = 2, label = 'Exchange [%s, %s]' % (h.getOrbitalName(n), h.getOrbitalName(n2)))
    i = i + 1
  plt.ylim((-ymin, ymin))
  plt.legend()
  plt.show()

