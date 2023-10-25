#pragma once

#include <string>
#include <nlohmann/json.hpp>

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key);
