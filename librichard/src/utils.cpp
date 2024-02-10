#include "richard/utils.hpp"
#include "richard/exception.hpp"
#include "richard/version.hpp"
#include <fstream>
#include <functional>

namespace richard {

hashedString_t hashString(const std::string& value) {
  static std::hash<std::string> hash;
  return hash(value);
}

uint32_t majorVersion() {
  return static_cast<uint32_t>(RICHARD_VERSION_MAJOR);
}

uint32_t minorVersion() {
  return static_cast<uint32_t>(RICHARD_VERSION_MINOR);
}

std::string versionString() {
  return STR(majorVersion() << "." << minorVersion());
}

std::ostream& operator<<(std::ostream& os, const Size3& size) {
  os << size[0] << ", " << size[1] << ", " << size[2];
  return os;
}

}
