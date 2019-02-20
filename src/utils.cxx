#include "utils.h"
#include <cmath>

ldouble F0(ldouble x) {
  return 0.5L * std::sqrt(M_PI/x) * std::erf(std::sqrt(x));
}

lm::lm(int li, int mi)
  : l(li), m(mi) {
}

lm::~lm() {
}

lm::lm(const lm &o)
  : l(o.l), m(o.m) {
}

lm &lm::operator =(const lm &o) {
  l = o.l;
  m = o.m;
  return *this;
}

bool lm::operator ==(const lm &o) const {
  return l == o.l && m == o.m;
}
bool lm::operator !=(const lm &o) const {
  return !(*this == o);
}

bool lm::operator >(const lm &o) const {
  if (l > o.l) return true;

  if (l == o.l) return m > o.l;
  
  return false;
}

bool lm::operator <(const lm &o) const {
  if (*this == o) return false;
  if (*this > o) return false;
  return true;
}

std::ostream &lm::operator <<(std::ostream &os) const {
  os << "(" << l << ", " << m << ")";
  return os;
}


double factorial(double n) {
  if (n == 1 || n == 0) return 1;
  return std::tgamma(n+1);
}

double CG(int j1, int j2, int m1, int m2, int j, int m) {
  if (std::abs(m) > j) return 0;
  if (j1 + j2 < j) return 0;
  if (std::abs(j1 - j2) > j) return 0;
  if (m1 + m2 != m) return 0;
  if (j1 < j2) return std::pow(-1.0, j - j1 - j2)*CG(j2, j1, m2, m1, j, m);
  if (m < 0) return std::pow(-1.0, j - j1 - j2)*CG(j1, j2, -m1, -m2, j, -m);
  if (j2 == 0) {
    if (j == j1 && m == m1) return 1;
    else return 0;
  } else if (j1 == 1 && j2 == 1) {
    if (m == 2) {
      if (m1 == 1 && m2 == 1 && j == 2) return 1;
      else return 0;
    } else if (m == 1) {
      if (m1 == 1 && m2 == 0 && j == 2)
        return std::pow(0.5, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 1)
        return std::pow(0.5, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 1)
        return -std::pow(0.5, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 2)
        return std::pow(0.5, 0.5);
      else return 0;
    } else if (m == 0) {
      if (m1 == 1 && m2 == -1 && j == 2)
        return std::pow(1.0/6.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 1)
        return std::pow(1.0/2.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 0)
        return std::pow(1.0/3.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 2)
        return std::pow(2.0/3.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 1)
        return 0;
      else if (m1 == 0 && m2 == 0 && j == 0)
        return - std::pow(1.0/3.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 2)
        return std::pow(1.0/6.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 1)
        return - std::pow(1.0/2.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 0)
        return std::pow(1.0/3.0, 0.5);
      else
        return 0;
    } else {
      return 0;
    }
  } else if (j1 == 2 && j2 == 1) {
    if (m == 3) {
      if (m1 == 2 && m2 == 1 && j == 3) return 1;
      else return 0;
    } else if (m == 2) {
      if (m1 == 2 && m2 == 0 && j == 3) return std::pow(1.0/3.0, 0.5);
      else if (m1 == 2 && m2 == 0 && j == 2) return std::pow(2.0/3.0, 0.5);
      else if (m1 == 1 && m2 == 1 && j == 3) return std::pow(2.0/3.0, 0.5);
      else if (m1 == 1 && m2 == 1 && j == 2) return - std::pow(1.0/3.0, 0.5);
      else return 0;
    } else if (m == 1) {
      if (m1 == 2 && m2 == -1 && j == 3) return std::pow(1.0/15.0, 0.5);
      else if (m1 == 2 && m2 == -1 && j == 2) return std::pow(1.0/3.0, 0.5);
      else if (m1 == 2 && m2 == -1 && j == 1) return std::pow(3.0/5.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 3) return std::pow(8.0/15.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 2) return std::pow(1.0/16.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 1) return -std::pow(3.0/10.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 3) return std::pow(2.0/5.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 2) return -std::pow(1.0/2.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 1) return std::pow(1.0/10.0, 0.5);
      else return 0;
    } else if (m == 0) {
      if (m1 == 1 && m2 == -1 && j == 3) return std::pow(1.0/5.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 2) return std::pow(1.0/2.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 1) return std::pow(3.0/10.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 3) return std::pow(3.0/5.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 2) return 0;
      else if (m1 == 0 && m2 == 0 && j == 1) return -std::pow(2.0/5.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 3) return std::pow(1.0/5.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 2) return -std::pow(1.0/2.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 1) return -std::pow(3.0/10.0, 0.5);
      else return 0;
    }
  } else if (j1 == 2 && j2 == 2) {
    if (m == 4) {
      if (m1 == 2 && m2 == 2 && j == 4) return 1;
      else return 0;
    } else if (m == 3) {
      if (m1 == 2 && m2 == 1 && j == 4) return std::pow(1.0/2.0, 0.5);
      else if (m1 == 2 && m2 == 1 && j == 3) return std::pow(1.0/2.0, 0.5);
      else if (m1 == 1 && m2 == 2 && j == 4) return std::pow(1.0/2.0, 0.5);
      else if (m1 == 1 && m2 == 2 && j == 3) return -std::pow(1.0/2.0, 0.5);
      else return 0;
    } else if (m == 2) {
      if (m1 == 2 && m2 == 0 && j == 4) return std::pow(3.0/14.0, 0.5);
      else if (m1 == 2 && m2 == 0 && j == 3) return std::pow(1.0/2.0, 0.5);
      else if (m1 == 2 && m2 == 0 && j == 3) return std::pow(1.0/2.0, 0.5);
      else if (m1 == 2 && m2 == 0 && j == 2) return std::pow(2.0/7.0, 0.5);
      else if (m1 == 1 && m2 == 1 && j == 4) return std::pow(4.0/7.0, 0.5);
      else if (m1 == 1 && m2 == 1 && j == 3) return 0;
      else if (m1 == 1 && m2 == 1 && j == 2) return -std::pow(3.0/7.0, 0.5);
      else if (m1 == 0 && m2 == 2 && j == 4) return std::pow(3.0/14.0, 0.5);
      else if (m1 == 0 && m2 == 2 && j == 3) return -std::pow(1.0/2.0, 0.5);
      else if (m1 == 0 && m2 == 2 && j == 2) return std::pow(2.0/7.0, 0.5);
      else return 0;
    } else if (m == 1) {
      if (m1 == 2 && m2 == -1 && j == 4) return std::pow(1.0/14.0, 0.5);
      else if (m1 == 2 && m2 == -1 && j == 3) return std::pow(3.0/10.0, 0.5);
      else if (m1 == 2 && m2 == -1 && j == 2) return std::pow(3.0/7.0, 0.5);
      else if (m1 == 2 && m2 == -1 && j == 1) return std::pow(1.0/5.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 4) return std::pow(3.0/7.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 3) return std::pow(1.0/5.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 2) return -std::pow(1.0/14.0, 0.5);
      else if (m1 == 1 && m2 == 0 && j == 1) return -std::pow(2.0/10.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 4) return std::pow(3.0/7.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 3) return -std::pow(1.0/5.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 2) return -std::pow(1.0/14.0, 0.5);
      else if (m1 == 0 && m2 == 1 && j == 1) return -std::pow(3.0/10.0, 0.5);
      else if (m1 == -1 && m2 == 2 && j == 4) return std::pow(1.0/14.0, 0.5);
      else if (m1 == -1 && m2 == 2 && j == 3) return -std::pow(3.0/10.0, 0.5);
      else if (m1 == -1 && m2 == 2 && j == 2) return std::pow(3.0/7.0, 0.5);
      else if (m1 == -1 && m2 == 2 && j == 1) return -std::pow(1.0/5.0, 0.5);
      else return 0;
    } else if (m == 0) {
      if (m1 == 2 && m2 == -2 && j == 4) return std::pow(1.0/70.0, 0.5);
      else if (m1 == 2 && m2 == -2 && j == 3) return std::pow(1.0/10.0, 0.5);
      else if (m1 == 2 && m2 == -2 && j == 2) return std::pow(2.0/7.0, 0.5);
      else if (m1 == 2 && m2 == -2 && j == 1) return std::pow(2.0/5.0, 0.5);
      else if (m1 == 2 && m2 == -2 && j == 0) return std::pow(1.0/5.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 4) return std::pow(8.0/35.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 3) return std::pow(2.0/5.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 2) return std::pow(1.0/14.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 1) return -std::pow(1.0/10.0, 0.5);
      else if (m1 == 1 && m2 == -1 && j == 0) return -std::pow(1.0/5.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 4) return -std::pow(18.0/35.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 3) return 0;
      else if (m1 == 0 && m2 == 0 && j == 2) return -std::pow(2.0/7.0, 0.5);
      else if (m1 == 0 && m2 == 0 && j == 1) return 0;
      else if (m1 == 0 && m2 == 0 && j == 0) return std::pow(1.0/5.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 4) return std::pow(8.0/35.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 3) return -std::pow(2.0/5.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 2) return std::pow(1.0/14.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 1) return std::pow(1.0/10.0, 0.5);
      else if (m1 == -1 && m2 == 1 && j == 0) return -std::pow(1.0/5.0, 0.5);
      else if (m1 == -2 && m2 == 2 && j == 4) return std::pow(1.0/70.0, 0.5);
      else if (m1 == -2 && m2 == 2 && j == 3) return -std::pow(1.0/10.0, 0.5);
      else if (m1 == -2 && m2 == 2 && j == 2) return std::pow(2.0/7.0, 0.5);
      else if (m1 == -2 && m2 == 2 && j == 1) return -std::pow(2.0/5.0, 0.5);
      else if (m1 == -2 && m2 == 2 && j == 0) return std::pow(1.0/5.0, 0.5);
      else return 0;
    }
  }

  return 0;
}

const std::string getSymbol(int Z) {
  static std::map<int, std::string> s = {
      {1, "H"},
      {2, "He"},
      {3, "Li"},
      {4, "Be"},
      {5, "B"},
      {6, "C"},
      {7, "N"},
      {8, "O"},
      {9, "F"},
      {10, "Ne"},
      {11, "Na"},
      {12, "Mg"},
      {13, "Al"},
      {14, "Si"},
      {15, "P"},
      {16, "S"},
      {17, "Cl"},
      {18, "Ar"},
      {19, "K"},
      {20, "Ca"},
      {21, "Sc"},
      {22, "Ti"},
      {23, "V"},
      {24, "Cr"},
      {25, "Mn"},
      {26, "Fe"},
      {27, "Co"},
      {28, "Ni"},
      {29, "Cu"},
      {30, "Zn"},
      {31, "Ga"},
      {32, "Ge"},
      {33, "As"},
      {34, "Se"},
      {35, "Br"},
      {36, "Kr"},
      {37, "Rb"},
      {38, "Sr"},
      {39, "Y"},
      {40, "Zr"},
      {41, "Nb"},
      {42, "Mo"},
      {43, "Tc"},
      {44, "Ru"},
      {45, "Rh"},
      {46, "Pd"},
      {47, "Ag"},
      {48, "Cd"},
      {49, "In"},
      {50, "Sn"},
      {51, "Sb"},
      {52, "Te"},
      {53, "I"},
      {54, "Xe"},
      {55, "Cs"},
      {56, "Ba"},
      {57, "La"},
      {58, "Ce"},
      {59, "Pr"},
      {60, "Nd"},
      {61, "Pm"},
      {62, "Sm"},
      {63, "Eu"},
      {64, "Gd"},
      {65, "Tb"},
      {66, "Dy"},
      {67, "Ho"},
      {68, "Er"},
      {69, "Tm"},
      {70, "Yb"},
      {71, "Lu"},
      {72, "Hf"},
      {73, "Ta"},
      {74, "W"},
      {75, "Re"},
      {76, "Os"},
      {77, "Ir"},
      {78, "Pt"},
      {79, "Au"},
      {80, "Hg"},
      {81, "Tl"},
      {82, "Pb"},
      {83, "Bi"},
      {84, "Po"},
      {85, "At"},
      {86, "Rn"},
      {87, "Fr"},
      {88, "Ra"},
      {89, "Ac"},
      {90, "Th"},
      {91, "Pa"},
      {92, "U"},
      {93, "Np"},
      {94, "Pu"},
      {95, "Am"},
      {96, "Cm"},
      {97, "Bk"},
      {98, "Cf"},
      {99, "Es"},
      {100, "Fm"},
      {101, "Md"},
      {102, "No"},
      {103, "Lr"},
      {104, "Rf"},
      {105, "Db"},
      {106, "Sg"},
      {107, "Bh"},
      {108, "Hs"},
      {109, "Mt"},
      {110, "Ds"},
      {111, "Rg"},
      {112, "Cn"},
      {113, "Nh"},
      {114, "Fl"},
      {115, "Mc"},
      {116, "Lv"},
      {117, "Ts"},
      {118, "Og"}
  };
  if (s.find(Z) != s.end()) {
    return s[Z];
  }
  return "Xxx";
}

