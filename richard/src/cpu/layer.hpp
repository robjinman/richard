#pragma once

#include "math.hpp"
#include "types.hpp"
#include <array>
#include <functional>
#include <cmath>

namespace richard {
namespace cpu {

using ActivationFn = std::function<netfloat_t(netfloat_t)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](netfloat_t x) {
  return static_cast<netfloat_t>(1.0 / (1.0 + exp(-x)));
};

const ActivationFn sigmoidPrime = [](netfloat_t x) {
  netfloat_t sigX = sigmoid(x);
  return static_cast<netfloat_t>(sigX * (1.0 - sigX));
};

const ActivationFn relu = [](netfloat_t x) {
  return x < 0.f ? 0.f : x;
};

const ActivationFn reluPrime = [](netfloat_t x) {
  return x < 0.f ? 0.f : 1.f;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
  const Vector& expected) {

  DBG_ASSERT(actual.size() == expected.size());
  return actual - expected;
};

class Layer {
  public:
    virtual Size3 outputSize() const = 0;
    virtual const DataArray& activations() const = 0;
    virtual const DataArray& inputDelta() const = 0;
    virtual void trainForward(const DataArray& inputs) = 0;
    virtual DataArray evalForward(const DataArray& inputs) const = 0;
    virtual void updateDeltas(const DataArray& inputs, const DataArray& outputDelta) = 0;
    virtual void updateParams(size_t epoch) = 0;
    virtual void writeToStream(std::ostream& stream) const = 0;

    virtual ~Layer() {}
};

using LayerPtr = std::unique_ptr<Layer>;

}
}
