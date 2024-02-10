#pragma once

#include "richard/neural_net.hpp"

namespace richard {

class EventSystem;
class Logger;
class FileSystem;
class PlatformPaths;
class Config;

namespace gpu {

NeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger);
NeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config, std::istream& stream,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger);

}
}
