#include "trace.hpp"
#include <iostream>

namespace richard {

Trace::Trace(const std::string& file, const std::string& func)
  : m_file(file)
  , m_func(func) {

  std::cout << "ENTER " << m_func << " (" << m_file << ")" << std::endl;
}

Trace::~Trace() {
  std::cout << "EXIT " << m_func << " (" << m_file << ")" << std::endl;
}

}
