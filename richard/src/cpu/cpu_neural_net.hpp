#pragma once

#include "neural_net.hpp"

class Logger;
class Layer;

class CpuNeuralNet : public NeuralNet {
  public:
    virtual Layer& getLayer(size_t idx) = 0;
    virtual ~CpuNeuralNet() = default;
};

using CpuNeuralNetPtr = std::unique_ptr<CpuNeuralNet>;

CpuNeuralNetPtr createCpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger);
CpuNeuralNetPtr createCpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger);

