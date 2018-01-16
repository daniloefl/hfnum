
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
fname = raw_input('')

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
  print "Energy for orbital %10s (index=%2d, n=%2d, l=%2d, m=%2d, s=%2d): %10.6f Hartree = %15.8f eV, first order LS correction: %10.6f Hartree = %15.8f eV" % (h.getOrbitalName(n), n, h.getOrbital_n(n), h.getOrbital_l(n), h.getOrbital_m(n), h.getOrbital_s(n), h.getOrbital_E(n), h.getOrbital_E(n)*eV, pert_E[n], pert_E[n]*eV)

python("Coefficients of orbitals:")
print coeff

# to implement
#E0 = h.getE0()
#print "Total ground energy: %10.6f Hartree = %15.8f eV" % (E0, E0*eV)

# show orbitals
f = plt.figure()
i = 0
for n in range(np.maximum(0, h.getNOrbitals()-len(style)), h.getNOrbitals()):
  plt.plot(r[:m], r[:m]*o[n][:m], style[i], linewidth = 2, label = 'r*R[%6s]' % (h.getOrbitalName(n)))
  i += 1
plt.legend()
plt.show()

