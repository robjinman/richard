#pragma once

#include <vector>
#include <array>
#include <nlohmann/json.hpp>
#include "types.hpp"

class NormalizationParams {
  public:
    NormalizationParams();
    explicit NormalizationParams(const nlohmann::json& json);

    double min;
    double max;
  
    static const nlohmann::json& exampleConfig();
};

inline double normalize(const NormalizationParams& params, double x) {
  return (x - params.min) / (params.max - params.min);
}

class DataDetails {
  public:
    DataDetails();
    explicit DataDetails(const nlohmann::json& json);

    NormalizationParams normalization;
    std::vector<std::string> classLabels;
    Triple shape;

    static const nlohmann::json& exampleConfig();
};

