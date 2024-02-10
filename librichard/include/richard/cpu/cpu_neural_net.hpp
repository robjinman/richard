#pragma once

#include "richard/neural_net.hpp"

namespace richard {

class EventSystem;
class Config;

namespace cpu {

class Layer;

class CpuNeuralNet : public NeuralNet {
  public:
    // For unit tests
    virtual Layer& test_getLayer(size_t idx) = 0;

    virtual ~CpuNeuralNet() = default;
};

using CpuNeuralNetPtr = std::unique_ptr<CpuNeuralNet>;

CpuNeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config,
  EventSystem& eventSystem);
CpuNeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config,
  std::istream& stream, EventSystem& eventSystem);

}
}
