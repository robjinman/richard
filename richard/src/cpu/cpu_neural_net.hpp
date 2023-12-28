#pragma once

#include "neural_net.hpp"

namespace richard {

class Logger;

namespace cpu {

class Layer;

class CpuNeuralNet : public NeuralNet {
  public:
    // For unit tests
    virtual Layer& test_getLayer(size_t idx) = 0;

    virtual ~CpuNeuralNet() = default;
};

using CpuNeuralNetPtr = std::unique_ptr<CpuNeuralNet>;

CpuNeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger);
CpuNeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger);

}
}
