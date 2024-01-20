#include "utils.hpp"
#include "exception.hpp"
#include <fstream>

namespace richard {

std::ostream& operator<<(std::ostream& os, const Size3& size) {
  os << size[0] << ", " << size[1] << ", " << size[2];
  return os;
}

}
