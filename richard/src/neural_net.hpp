#pragma once

#include <vector>
#include "math.hpp"

struct TrainingSample {
  char label;
  Vector data;
};

struct Layer {
  Matrix weights;
  Vector biases;

  Layer(Layer&& mv);
  Layer(Matrix&& weights, Vector&& biases);
};

class NeuralNet {
  public:
    NeuralNet(size_t inputs, std::initializer_list<size_t> layers, size_t outputs);

    void train(const TrainingSample& sample);
    Vector evaluate(const Vector& inputs) const;

  private:
    Vector m_inputs;
    std::vector<Layer> m_layers;
};
