#pragma once

#include <nlohmann/json.hpp>
#include <string>

nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key);

