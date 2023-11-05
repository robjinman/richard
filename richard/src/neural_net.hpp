#pragma once

#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include "math.hpp"
#include "types.hpp"

class Logger;
class LabelledDataSet;
class Layer;

class NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    virtual CostFn costFn() const = 0;
    virtual Triple inputSize() const = 0;
    virtual void writeToStream(std::ostream& s) const = 0;
    virtual void train(LabelledDataSet& data) = 0;
    virtual VectorPtr evaluate(const Array3& inputs) const = 0;

    // Called from another thread
    virtual void abort() = 0;

    static const nlohmann::json& exampleConfig();

    virtual ~NeuralNet() {}

    // Exposed for testing
    //
    virtual Layer& getLayer(size_t index) = 0;
};

using NeuralNetPtr = std::unique_ptr<NeuralNet>;

NeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger);
NeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger);

