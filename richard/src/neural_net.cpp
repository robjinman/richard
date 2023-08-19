#include <cmath>
#include "neural_net.hpp"

namespace {

using ActivationFn = std::function<double(double)>;

const ActivationFn sigmoid = [](double x) {
  return 1.0 / (1.0 + exp(-x));
};

}

Layer::Layer(Layer&& mv)
  : weights(std::move(mv.weights))
  , biases(std::move(mv.biases)) {}

Layer::Layer(Matrix&& weights, Vector&& biases)
  : weights(std::move(weights))
  , biases(std::move(biases)) {}

NeuralNet::NeuralNet(size_t inputs, std::initializer_list<size_t> layers, size_t outputs)
  : m_inputs(inputs) {

  size_t prevLayerSize = inputs;
  for (size_t layerSize : layers) {
    m_layers.push_back(Layer(Matrix(prevLayerSize, layerSize), Vector(layerSize)));

    prevLayerSize = layerSize;
  }

  m_layers.push_back(Layer(Matrix(prevLayerSize, outputs), Vector(outputs)));
}

void NeuralNet::train(const TrainingSample& sample) {
  // TODO
}

Vector NeuralNet::evaluate(const Vector& inputs) const {
  Vector activations(inputs);

  for (const auto& layer : m_layers) {
    activations = (layer.weights * activations + layer.biases).transform(sigmoid);
  }

  return activations;
}
