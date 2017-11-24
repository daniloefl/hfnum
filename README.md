# hfnum

Hartree-Fock calculation in C++ using a numerical Grid. Based on hfpython repository.

# Installing packages necessary for compilation

```
sudo apt install libboost-dev* libboost-python*
```

# Compilation

```
cmake .
make
```

# Running

Try getting the energy and orbital shapes for Hydrogen with:

```
python share/test_H.py
```

For Helium, with:


```
python share/test_He.py
```

