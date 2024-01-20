#pragma once

#include "types.hpp"
#include "config.hpp"
#include <vector>
#include <array>

namespace richard {

class NormalizationParams {
  public:
    NormalizationParams();
    explicit NormalizationParams(const Config& config);

    netfloat_t min;
    netfloat_t max;
  
    static const Config& exampleConfig();
};

inline netfloat_t normalize(const NormalizationParams& params, netfloat_t x) {
  return (x - params.min) / (params.max - params.min);
}

class DataDetails {
  public:
    DataDetails();
    explicit DataDetails(const Config& config);

    static const Config& exampleConfig();

    NormalizationParams normalization;
    std::vector<std::string> classLabels;
    Size3 shape;
    size_t batchSize;
};

}
