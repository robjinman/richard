#include "data_details.hpp"
#include "util.hpp"

NormalizationParams::NormalizationParams(const nlohmann::json& json)
  : min(getOrThrow(json, "min").get<double>())
  , max(getOrThrow(json, "max").get<double>()) {}

DataDetails::DataDetails(const nlohmann::json& json)
  : normalization(getOrThrow(json, "normalization"))
  , classLabels(getOrThrow(json, "classes").get<std::vector<std::string>>())
  , shape(getOrThrow(json, "shape").get<Triple>()) {}

