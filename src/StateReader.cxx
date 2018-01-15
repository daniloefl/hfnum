#include "StateReader.h"
#include <string>
#include <fstream>
#include "utils.h"
#include <iostream>

StateReader::StateReader() {
}

StateReader::StateReader(const std::string &fin) {
  load(fin);
}

StateReader::~StateReader() {
  for (int k = 0; k < _o.size(); ++k) {
    delete _o[k];
  }
  _o.clear();
}

Orbital *StateReader::getOrbital(const int i) {
  return _o[i];
}

void StateReader::load(const std::string &fin) {
  std::ifstream f(fin.c_str());
  std::string line;
  std::string trash;

  while(std::getline(f, line)) {
    std::stringstream ss;
    ss.str(line);

    std::string mode;
    int a_i;
    ldouble a_d;
    std::vector<ldouble> a_v;

    ss >> mode;
    if (mode == "method") {
      ss >> a_i;
      _i[mode] = a_i;
    } else if (mode == "Z") {
      ss >> a_d;
      _d[mode] = a_d;
    } else if (mode == "gamma_scf") {
      ss >> a_d;
      _d[mode] = a_d;
    } else if (mode == "central") {
      ss >> trash;
    } else if (mode == "grid.isLog") {
      ss >> a_i;
      _i[mode] = a_i;
    } else if (mode == "grid.dx") {
      ss >> a_d;
      _d[mode] = a_d;
    } else if (mode == "grid.N") {
      ss >> a_i;
      _i[mode] = a_i;
    } else if (mode == "grid.rmin") {
      ss >> a_d;
      _d[mode] = a_d;
    } else if (mode == "orbital") {
      int io;
      ss >> io;

      std::string trash;

      int o_N, o_L, o_M, o_S;
      ss >> trash >> o_N >> trash >> o_L >> trash >> o_M >> trash >> o_S;

      ldouble o_E;
      ss >> trash >> o_E;

      _o.push_back(new Orbital(o_S, o_N, o_L, o_M));
      int k = _o.size()-1;
      _o[k]->N(_i["grid.N"]);
      _o[k]->E(o_E);
      int sphSize = 1;
      ss >> trash >> sphSize;
      for (int idx = 0; idx < sphSize; ++idx) {
        int l;
        int m;
        ss >> trash >> l >> trash >> m;

        if (l != o_L && m != o_M) _o[k]->addSphHarm(l, m);

        ss >> trash;

        ldouble read_value;
        for (int ir = 0; ir < _i["grid.N"]; ++ir) { // for each radial point
          ss >> read_value;
          (*_o[k])(ir, l, m) = read_value;
        }
      }
    } else if (mode == "vd") {
      int io;
      ss >> io;

      std::string trash;

      int v_l, v_m;
      ss >> trash >> v_l >> trash >> v_m;

      ss >> trash;

      _vd[io] = std::vector<ldouble>(_i["grid.N"], 0);
      ldouble read_value;
      for (int k = 0; k < _i["grid.N"]; ++k) {
        ss >> read_value;
        _vd[io][k] = read_value;
      }
    } else if (mode == "vex") {
      int io1, io2;
      ss >> io1 >> io2;

      std::string trash;

      int v_l, v_m;
      ss >> trash >> v_l >> trash >> v_m;

      ss >> trash;

      _vex[std::pair<int, int>(io1, io2)] = std::vector<ldouble>(_i["grid.N"], 0);
      ldouble read_value;
      for (int k = 0; k < _i["grid.N"]; ++k) {
        ss >> read_value;
        _vex[std::pair<int, int>(io1, io2)][k] = read_value;
      }
    }
  }
}

std::vector<ldouble> &StateReader::getVector(const std::string &id) {
  return _v[id];
}

std::vector<ldouble> &StateReader::getVd(const int i) {
  return _vd[i];
}

std::vector<ldouble> &StateReader::getVex(const int i, const int j) {
  return _vex[std::pair<int,int>(i, j)];
}

ldouble StateReader::getDouble(const std::string &id) {
  return _d[id];
}

int StateReader::getInt(const std::string &id) {
  return _i[id];
}
