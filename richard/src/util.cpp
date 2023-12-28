#include "util.hpp"
#include "exception.hpp"
#include <fstream>

namespace richard {

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key){
  if (!obj.contains(key)) {
    EXCEPTION("Expected '" << key << "' key in JSON object");
  }
  return obj[key];
}

std::string loadFile(const std::string& path) {
  std::ifstream stream(path);
  std::stringstream ss;
  std::string line;
  while (std::getline(stream, line)) {
    ss << line << std::endl;
  }
  return ss.str();
}

size_t tripleProduct(const Triple& t) {
  return t[0] * t[1] * t[2];
}

size_t tripleSum(const Triple& t) {
  return t[0] + t[1] + t[2];
}

}
