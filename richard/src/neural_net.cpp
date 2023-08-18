#include <functional>
#include <cmath>
#include "neural_net.hpp"

namespace {

using ActivationFn = std::function<double(double)>;

const ActivationFn sigmoid = [](double x) {
  return 1.0 / (1.0 + exp(-x));
};

}

NeuralNet::NeuralNet(size_t inputs, std::initializer_list<size_t> layers, size_t outputs)
  : m_inputs(inputs)
  , m_outputs(outputs) {
  
  for (size_t layer : layers) {
    m_layers.push_back(Vector(layer));
  }
}

void NeuralNet::train(const TrainingSample& sample) {
  
}

const Vector& NeuralNet::evaluate(const Vector& inputs) const {
  // TODO

  return m_outputs;
}
