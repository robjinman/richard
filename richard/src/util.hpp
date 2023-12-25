#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <sstream>

namespace richard {

#define STR(x) (std::stringstream("") << x).str()

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key);

}
