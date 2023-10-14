#pragma once

#include <array>
#include <functional>
#include "math.hpp"

using ActivationFn = std::function<double(double)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](double x) -> double {
  return 1.0 / (1.0 + exp(-x));
};

const ActivationFn sigmoidPrime = [](double x) -> double {
  double sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
};

const ActivationFn relu = [](double x) -> double {
  return x < 0.0 ? 0.0 : x;
};

const ActivationFn reluPrime = [](double x) -> double {
  return x < 0.0 ? 0.0 : 1.0;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
                                                      const Vector& expected) -> Vector {
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
    virtual const Vector& activations() const = 0;
    virtual const Vector& delta() const = 0;
    virtual void trainForward(const Vector& inputs) = 0;
    virtual Vector evalForward(const Vector& inputs) const = 0;
    virtual void updateDelta(const Vector& layerInputs, const Layer& nextLayer, size_t epoch) = 0;
    virtual nlohmann::json getConfig() const = 0;
    virtual void writeToStream(std::ostream& fout) const = 0;
    virtual const Matrix& W() const = 0;

    virtual ~Layer() {}
};

std::ostream& operator<<(std::ostream& os, LayerType layerType);
