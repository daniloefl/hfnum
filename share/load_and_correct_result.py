#!/usr/bin/env python3

import sys
sys.path.append("../lib/")
sys.path.append("lib/")

eV = 27.21138602

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
fname = input('')

print "Loading result from file %s" % fname

# random initialisation
h = hfnum.NonCentralCorrection()
h.load(fname)

r = np.asarray(h.getR())

print("Correcting it due to central potential assumption.")
h.correct()

pert_E = h.getCorrectedE()
coeff = h.getCoefficients()

for n in range(0, h.getNOrbitals()):
  print "Energy for orbital %10s: %10.6f Hartree = %15.8f eV, first order LS correction: %10.6f Hartree = %15.8f eV" % (h.getOrbitalName(n), h.getOrbital_E(n), h.getOrbital_E(n)*eV, pert_E[n], pert_E[n]*eV)

print "Coefficients of orbitals:"
print coeff

E0 = h.getE0()
E0uncorr = h.getE0Uncorrected()
print "Total ground energy -- before correction: %10.6f Hartree = %15.8f eV, after correction: %10.6f Hartree = %15.8f eV" % (E0uncorr, E0uncorr*eV, E0, E0*eV)

m = next(i for i,v in enumerate(r) if v >= 10)
o = []
for n in range(0, h.getNOrbitals()):
  o.append(np.asarray(h.getCentral(n)))

style = ['r-', 'b-', 'm-', 'c-', 'k-', 'g-']

# show orbitals
f = plt.figure()
i = 0
for n in range(np.maximum(0, h.getNOrbitals()-len(style)), h.getNOrbitals()):
  plt.plot(r[:m], r[:m]*o[n][:m], style[i], linewidth = 2, label = 'r*R[%6s]' % (h.getOrbitalName(n)))
  i += 1
plt.legend()
plt.show()

