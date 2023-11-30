#pragma once

#include "math.hpp"
#include "types.hpp"
#include <array>
#include <functional>
#include <cmath>

using ActivationFn = std::function<netfloat_t(netfloat_t)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](netfloat_t x) {
  return 1.0 / (1.0 + exp(-x));
};

const ActivationFn sigmoidPrime = [](netfloat_t x) {
  netfloat_t sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
};

const ActivationFn relu = [](netfloat_t x) {
  return x < 0.0 ? 0.0 : x;
};

const ActivationFn reluPrime = [](netfloat_t x) {
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
    virtual Triple outputSize() const = 0;
    virtual const DataArray& activations() const = 0;
    virtual const DataArray& delta() const = 0;
    virtual void trainForward(const DataArray& inputs) = 0;
    virtual DataArray evalForward(const DataArray& inputs) const = 0;
    virtual void updateDelta(const DataArray& inputs, const Layer& nextLayer) = 0;
    virtual void updateParams(size_t epoch) = 0;
    virtual void writeToStream(std::ostream& fout) const = 0;
    virtual const Matrix& W() const = 0;

    virtual ~Layer() {}
};

std::ostream& operator<<(std::ostream& os, LayerType layerType);

