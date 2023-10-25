#pragma once

#include <stdexcept>
#include <sstream>
#include <cassert>

#define ASSERT(X) assert(X)

#define EXCEPTION(msg) \
{ \
  std::stringstream ss; \
  ss << msg; \
  throw Exception(ss.str(), __FILE__, __LINE__); \
}

#define TRUE_OR_THROW(condition, msg) \
{ \
  if (!(condition)) { \
    EXCEPTION(msg) \
  } \
}

class Exception : public std::runtime_error {
  public:
    Exception(const std::string& msg, const std::string& file, int line)
      : runtime_error(msg + " (" + file + ", " + std::to_string(line) + ")") {}
};
