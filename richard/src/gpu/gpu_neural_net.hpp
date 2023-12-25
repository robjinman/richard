#pragma once

#include "neural_net.hpp"

namespace richard {

class Logger;

namespace gpu {

NeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger);
NeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger);

}
}
