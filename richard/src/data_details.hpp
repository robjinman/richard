#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include <array>

namespace richard {

class NormalizationParams {
  public:
    NormalizationParams();
    explicit NormalizationParams(const nlohmann::json& json);

    netfloat_t min;
    netfloat_t max;
  
    static const nlohmann::json& exampleConfig();
};

inline netfloat_t normalize(const NormalizationParams& params, netfloat_t x) {
  return (x - params.min) / (params.max - params.min);
}

class DataDetails {
  public:
    DataDetails();
    explicit DataDetails(const nlohmann::json& json);

    NormalizationParams normalization;
    std::vector<std::string> classLabels;
    Triple shape;
    size_t batchSize;

    static const nlohmann::json& exampleConfig();
};

}
