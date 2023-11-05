#pragma once

#include <stdexcept>
#include <sstream>

// DBG_ macros expand to nothing in release builds. Use in performance critical areas.

#define EXCEPTION(msg) \
{ \
  std::stringstream ss; \
  ss << msg; \
  throw Exception(ss.str(), __FILE__, __LINE__); \
}

#define ASSERT_MSG(condition, msg) \
{ \
  if (!(condition)) { \
    EXCEPTION(msg) \
  } \
}

#define ASSERT(X) ASSERT_MSG(X, "Assertion failed: " #X)

#ifdef NDEBUG
  #define DBG_ASSERT(X)
  #define DBG_ASSERT_MSG(X, msg)
#else
  #define DBG_ASSERT(X) ASSERT(X)
  #define DBG_ASSERT_MSG(X, msg) ASSERT_MSG(X, msg)
#endif

class Exception : public std::runtime_error {
  public:
    Exception(const std::string& msg, const std::string& file, int line)
      : runtime_error(msg + " (" + file + ", " + std::to_string(line) + ")") {}
};

