#pragma once

#include <string>

namespace richard {

class Trace {
  public:
    Trace(const std::string& file, const std::string& func);
    ~Trace();

  private:
    std::string m_file;
    std::string m_func;
};

#ifdef TRACE
  #define DBG_TRACE Trace t(__FILE__, __FUNCTION__);
#else
  #define DBG_TRACE
#endif

}
