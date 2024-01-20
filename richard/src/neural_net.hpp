#pragma once

#include "math.hpp"
#include "types.hpp"

namespace richard {

class Config;
class LabelledDataSet;

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const Config& obj);

  size_t epochs;
  size_t batchSize;
  size_t miniBatchSize;

  static const Config& exampleConfig();
};

class NeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    virtual CostFn costFn() const = 0;
    virtual Size3 inputSize() const = 0;
    virtual void writeToStream(std::ostream& stream) const = 0;
    virtual void train(LabelledDataSet& data) = 0;
    virtual VectorPtr evaluate(const Array3& inputs) const = 0;

    // Called from another thread
    virtual void abort() = 0;

    static const Config& exampleConfig();

    virtual ~NeuralNet() {}
};

using NeuralNetPtr = std::unique_ptr<NeuralNet>;

}
