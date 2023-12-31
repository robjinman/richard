#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <sstream>

namespace richard {

#define STR(x) (std::stringstream("") << x).str()

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key);
std::string loadFile(const std::string& path);

constexpr size_t calcProduct(const Size3& s) {
  return s[0] * s[1] * s[2];
}

constexpr size_t calcSum(const Size3& s) {
  return s[0] + s[1] + s[2];
}

}
