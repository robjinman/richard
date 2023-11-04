#include "data_details.hpp"
#include "util.hpp"

NormalizationParams::NormalizationParams()
  : min(0)
  , max(0) {}

NormalizationParams::NormalizationParams(const nlohmann::json& json)
  : min(getOrThrow(json, "min").get<double>())
  , max(getOrThrow(json, "max").get<double>()) {}

const nlohmann::json& NormalizationParams::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;

  if (!done) {
    obj["min"] = 0;
    obj["max"] = 255;
    
    done = true;
  }

  return obj;
}

DataDetails::DataDetails(const nlohmann::json& json)
  : normalization(getOrThrow(json, "normalization"))
  , classLabels(getOrThrow(json, "classes").get<std::vector<std::string>>())
  , shape(getOrThrow(json, "shape").get<Triple>()) {}

const nlohmann::json& DataDetails::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;

  if (!done) {
    obj["normalization"] = NormalizationParams::exampleConfig();
    obj["classes"] = std::vector<std::string>({
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    });
    obj["shape"] = Triple({ 28, 28, 1 });

    done = true;
  }

  return obj;
}

