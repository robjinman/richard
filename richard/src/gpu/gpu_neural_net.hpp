#pragma once

#include "neural_net.hpp"

namespace richard {

class Logger;
class FileSystem;
class PlatformPaths;
class Config;

namespace gpu {

NeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config, FileSystem& fileSystem,
  const PlatformPaths& platformPaths, Logger& logger);
NeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config, std::istream& stream,
  FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger);

}
}
