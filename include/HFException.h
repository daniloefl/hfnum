#ifndef HFEXCEPTION_H
#define HFEXCEPTION_H

#include <exception>
#include <string>

class HFException : public std::exception {
  protected:
    std::string _msg;
  public:
    HFException(const std::string &msg = "");
    ~HFException() throw();

    const char *what() const throw();

};

#endif

