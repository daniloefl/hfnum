
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

Z = raw_input('[default: %d] ' % 1)
if Z.strip() == '':
  Z = 1
Z = float(Z) ## allow for effective atomic numbers

h = hfnum.HF(Z)

config = ""
while config.strip() == "":
  print("Please enter the electron configuration in the format: 1s2 2s2 2p3")
  config = raw_input(" ")
  if config.strip() == "":
    print("Configuration %s does not match the allowed electron configuration format." % i)
    print("Please enter a valid electron configuration.")
    print("The format is a list of elements separated by spaces, where each element has the syntax: [principal number][angular momentum sub-shell][electron multiplicity].")
    print("[principal number] is a positive integer.")
    print("[angular momentum sub-shell] is one of s, p, d, f, g, h corresponding to l = 0, 1, 2, 3, 4, 5.")
    print("[electron multiplicity] is a positive integer not greater than 2*l + 1.")
    print("Example: 1s2 2s2 2p6.")
    continue
  config_split = config.split()
  r = re.compile("([0-9]+)([spdfghSPDFGH]+)([0-9]+)")
  orb = []
  electronCount = 0
  for i in config_split:
    m = r.match(i)
    if m == None:
      print("Configuration %s does not match the allowed electron configuration format." % i)
      print("Please enter a valid electron configuration.")
      print("The format is a list of elements separated by spaces, where each element has the syntax: [principal number][angular momentum sub-shell][electron multiplicity].")
      print("[principal number] is a positive integer.")
      print("[angular momentum sub-shell] is one of s, p, d, f, g, h corresponding to l = 0, 1, 2, 3, 4, 5.")
      print("[electron multiplicity] is a positive integer not greater than 2*l + 1.")
      print("Example: 1s2 2s2 2p6.")
      config = ""
      break
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
    electronCount += mult

symbol_name = "Xx" # unknown
charge = Z - electronCount
if Z.is_integer():
  symbol_name = hfnum.getSymbol(int(Z))
  charge = int(charge)
  symbol_name += "%d" % charge

print(">>> Z = %f, electron count = %d, symbol = %s" % (Z, electronCount, symbol_name))

print("Please enter the output file name.")
import readline, glob
def complete(text, state):
  return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)
fname_default = "calculate_HF_output/results_%s.txt" % symbol_name
fname = fname_default
fname = raw_input('[default: %s] ' % fname_default)
if fname.strip() == '':
    fname = fname_default

NiterSCF = 40
Niter = 100
F0stop = 1e-6
r = np.asarray(h.getR())
print("Last r:", r[-1])
print("First r:", r[0])
print("Number of grid points: ", len(r))
h.gammaSCF(0.4)
h.solve(NiterSCF, Niter, F0stop)

for n in range(0, h.getNOrbitals()):
  print("Energy for orbital %10s: %10.6f Hartree = %15.8f eV" % (h.getOrbitalName(n), h.getOrbital_E(n), h.getOrbital_E(n)*hfnum.eV))

E0 = h.getE0()
print("Total ground energy: %10.6f Hartree = %15.8f eV" % (E0, E0*hfnum.eV))

h.save(fname)

