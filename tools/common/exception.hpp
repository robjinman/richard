#pragma once

#include <sstream>
#include <stdexcept>

#define EXCEPTION(msg) { \
  std::stringstream ss; \
  ss << msg; \
  throw std::runtime_error(ss.str()); \
}
