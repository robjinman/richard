#pragma once

#include <cassert>
#include <string>
#include <nlohmann/json.hpp>

#define ASSERT(X) assert(X)

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key);
