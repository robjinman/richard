#pragma once

#include <vector>
#include "math.hpp"

class NeuralNet {
  public:
    NeuralNet(size_t inputs, std::initializer_list<size_t> layers, size_t outputs);

    const Vector& evaluate(const Vector& inputs) const;

  private:
    Vector m_inputs;
    std::vector<Vector> m_layers;
    Vector m_outputs;
};
