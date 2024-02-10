#pragma once

#include "richard/math.hpp"
#include "richard/types.hpp"
#include "richard/event_system.hpp"
#include <vector>

namespace richard {

class Config;
class LabelledDataSet;

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const Config& obj);

  uint32_t epochs;
  uint32_t batchSize;
  uint32_t miniBatchSize;

  static const Config& exampleConfig();
};

struct ESampleProcessed : public Event {
  ESampleProcessed(uint32_t sample, uint32_t samples)
    : Event(name)
    , sample(sample)
    , samples(samples) {}

  uint32_t sample;
  uint32_t samples;

  static const hashedString_t name;
};

struct EEpochStarted : public Event {
  EEpochStarted(uint32_t epoch, uint32_t epochs)
    : Event(name)
    , epoch(epoch)
    , epochs(epochs) {}

  uint32_t epoch;
  uint32_t epochs;

  static const hashedString_t name;
};

struct EEpochCompleted : public Event {
  EEpochCompleted(uint32_t epoch, uint32_t epochs, netfloat_t cost)
    : Event(name)
    , epoch(epoch)
    , epochs(epochs)
    , cost(cost) {}

  uint32_t epoch;
  uint32_t epochs;
  netfloat_t cost;

  static const hashedString_t name;
};

using ModelDetails = std::vector<std::pair<std::string, std::string>>;

class NeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    virtual CostFn costFn() const = 0;
    virtual Size3 inputSize() const = 0;
    virtual void writeToStream(std::ostream& stream) const = 0;
    virtual void train(LabelledDataSet& data) = 0;
    virtual Vector evaluate(const Array3& inputs) const = 0;
    virtual ModelDetails modelDetails() const = 0;

    // Called from another thread
    virtual void abort() = 0;

    static const Config& exampleConfig();

    virtual ~NeuralNet() {}
};

using NeuralNetPtr = std::unique_ptr<NeuralNet>;

}
