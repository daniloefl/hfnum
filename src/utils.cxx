#include "utils.h"
#include <cmath>

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

