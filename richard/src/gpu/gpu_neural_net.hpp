#pragma once

#include "neural_net.hpp"

namespace richard {

class Logger;
class FileSystem;
class PlatformPaths;

namespace gpu {

NeuralNetPtr createNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger);
NeuralNetPtr createNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  std::istream& stream, FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger);

}
}
