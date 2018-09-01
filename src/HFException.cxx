#include "HFException.h"

HFException::HFException(const std::string &msg)
  : _msg(msg) {
}

HFException::~HFException() throw() {
}

const char * HFException::what() const throw() {
  return _msg.c_str();
} 
