#pragma once

#include <vector>
#include <array>
#include <nlohmann/json.hpp>
#include "types.hpp"

struct NormalizationParams {
  NormalizationParams()
    : min(0)
    , max(0) {}

  NormalizationParams(const nlohmann::json& json);

  double min;
  double max;
};

inline double normalize(const NormalizationParams& params, double x) {
  return (x - params.min) / (params.max - params.min);
}

class DataDetails {
  public:
    DataDetails(const nlohmann::json& json);
    
    NormalizationParams normalization;
    std::vector<std::string> classLabels;
    Triple shape;
};

