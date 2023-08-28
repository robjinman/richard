#include <fstream>
#include <sstream>
#include "util.hpp"

std::map<std::string, std::string> readKeyValuePairs(std::istream& fin) {
  std::map<std::string, std::string> keyVals;

  std::string line;
  while (std::getline(fin, line)) {
    std::stringstream ss(line);
    std::string key, value;

    std::getline(ss, key, '=');
    std::getline(ss, value);

    if (!key.empty() && !value.empty()) {
      keyVals[key] = value;
    }
  }

  return keyVals;
}
