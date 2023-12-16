#pragma once

#include "neural_net.hpp"

class Logger;

NeuralNetPtr createGpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger);
NeuralNetPtr createGpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger);

