#pragma once

#include <array>
#include <functional>
#include "math.hpp"
#include "util.hpp"

using ActivationFn = std::function<double(double)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](double x) {
  return 1.0 / (1.0 + exp(-x));
};

const ActivationFn sigmoidPrime = [](double x) {
  double sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
};

const ActivationFn relu = [](double x) {
  return x < 0.0 ? 0.0 : x;
};

const ActivationFn reluPrime = [](double x) {
  return x < 0.0 ? 0.0 : 1.0;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
  const Vector& expected) {

  ASSERT(actual.size() == expected.size());
  return actual - expected;
};

enum class LayerType {
  DENSE,
  CONVOLUTIONAL,
  MAX_POOLING,
  OUTPUT
};

class Layer {
  public:
    virtual LayerType type() const = 0;
    virtual std::array<size_t, 3> outputSize() const = 0;
    virtual const DataArray& activations() const = 0;
    virtual const DataArray& delta() const = 0;
    virtual void trainForward(const DataArray& inputs) = 0;
    virtual DataArray evalForward(const DataArray& inputs) const = 0;
    virtual void updateDelta(const DataArray& inputs, const Layer& nextLayer, size_t epoch) = 0;
    virtual nlohmann::json getConfig() const = 0;
    virtual void writeToStream(std::ostream& fout) const = 0;
    virtual const Matrix& W() const = 0;

    virtual ~Layer() {}

    // Exposed for testing
    //
    virtual void setWeights(const Matrix& weights) = 0;
    virtual void setBiases(const Vector& biases) = 0;
};

std::ostream& operator<<(std::ostream& os, LayerType layerType);
