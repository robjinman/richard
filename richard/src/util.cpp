#include "util.hpp"
#include "exception.hpp"
#include <fstream>

namespace richard {

// TODO: Turn into function template that calls get<T>()
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

}
