#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include <array>

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
    size_t batchSize;

    static const nlohmann::json& exampleConfig();
};

