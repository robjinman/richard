#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <sstream>

namespace richard {

#define STR(x) (std::stringstream("") << x).str()

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key);
std::string loadFile(const std::string& path);
size_t tripleProduct(const Triple& t);
size_t tripleSum(const Triple& t);

}
