#include <cmath>
#include <iostream> // TODO
#include "neural_net.hpp"
#include "util.hpp"

namespace {

using ActivationFn = std::function<double(double)>;
using CostFn = std::function<double(const Vector&, const Vector&)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](double x) -> double {
  return 1.0 / (1.0 + exp(-x));
};

const ActivationFn sigmoidPrime = [](double x) -> double {
  double sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
};

const CostFn meanSquaredError = [](const Vector& actual, const Vector& expected) {
  ASSERT(actual.size() == expected.size());

  Vector diff = expected - actual;
  return diff.hadamard(diff).sum() / diff.size();
};

const CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  ASSERT(actual.size() == expected.size());

  Vector diff = expected - actual;
  return diff.hadamard(diff).sum() / 2.0;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
                                                      const Vector& expected) -> Vector {
  ASSERT(actual.size() == expected.size());

  Vector tmp = actual - expected;
  return tmp;
};

}

TrainingData::TrainingData(const std::vector<char>& labels)
  : m_labels(labels) {

  for (size_t i = 0; i < m_labels.size(); ++i) {
    Vector v(m_labels.size());
    v.zero();
    v[i] = 1.0;
    m_classOutputVectors.insert({m_labels[i], v});
  }
}
/*
NeuralNet::Layer::Layer(Layer&& mv)
  : weights(std::move(mv.weights))
  , biases(std::move(mv.biases))
  , Z(std::move(mv.Z)) {}

NeuralNet::Layer::Layer(Matrix&& weights, Vector&& biases)
  : weights(std::move(weights))
  , biases(std::move(biases))
  , Z(1) {}
*/
NeuralNet::Layer::Layer(const Layer& cpy)
  : weights(cpy.weights)
  , biases(cpy.biases)
  , Z(cpy.Z) {}

NeuralNet::Layer::Layer(const Matrix& weights, const Vector& biases)
  : weights(weights)
  , biases(biases)
  , Z(1) {}

NeuralNet::NeuralNet(std::initializer_list<size_t> layers) {
  size_t prevLayerSize = 0;
  size_t i = 0;
  for (size_t layerSize : layers) {
    if (i == 0) {
      prevLayerSize = layerSize;
      ++i;
      continue;
    }

    Matrix weights(prevLayerSize, layerSize);
    weights.randomize(1.0);

    Vector biases(layerSize);
    biases.randomize(1.0);

    //m_layers.push_back(Layer(std::move(weights), std::move(biases)));
    m_layers.emplace_back(weights, biases);

    prevLayerSize = layerSize;
    ++i;
  }
}

void NeuralNet::setWeights(const std::vector<Matrix>& W) {
  if (W.size() != m_layers.size()) {
    throw std::runtime_error("Wrong number of weight matrices");
  }

  for (size_t i = 0; i < W.size(); ++i) {
    m_layers[i].weights = W[i];
  }
}

void NeuralNet::setBiases(const std::vector<Vector>& B) {
  if (B.size() != m_layers.size()) {
    throw std::runtime_error("Wrong number of bias vectors");
  }

  for (size_t i = 0; i < B.size(); ++i) {
    m_layers[i].biases = B[i];
  }
}

void NeuralNet::feedForward(const Vector& x) {
  Vector A(1);

  size_t i = 0;
  for (Layer& layer : m_layers) {
    layer.Z = layer.weights * (i == 0 ? x : A) + layer.biases;
    A = layer.Z.transform(sigmoid);

    ++i;
  }
}

void NeuralNet::updateLayer(size_t layerIdx, const Vector& delta, const Vector& x,
  double learnRate) {

  Layer& layer = m_layers[layerIdx];

  Vector prevLayerActivations = layerIdx == 0 ? x : m_layers[layerIdx - 1].Z.transform(sigmoid);

  //std::cout << "weights size: " << layer.weights.rows() << ", " << layer.weights.cols() << "\n";
  for (size_t j = 0; j < layer.weights.rows(); j++) {
    for (size_t k = 0; k < layer.weights.cols(); k++) {
      double dw = prevLayerActivations[k] * delta[j] * learnRate;

      //std::cout << "dw = " << dw << "\n";

      double w = layer.weights.at(k, j);
      layer.weights.set(k, j, w - dw);
    }
  }

  layer.biases = layer.biases - delta * learnRate;
}

void NeuralNet::train(const TrainingData& data) {
  const std::vector<TrainingData::Sample>& samples = data.data();
  const size_t epochs = 100;
  double learnRate = 0.1;
  double learnRateDecay = 0.95;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (const auto& sample : samples) {
      const Vector& x = sample.data;
      const Vector& y = data.classOutputVector(sample.label);

      feedForward(x);

      Layer& outputLayer = m_layers.back();
      const Vector& Z = outputLayer.Z;
      Vector A = Z.transform(sigmoid);

      Vector deltaC = quadraticCostDerivatives(A, y);
      Vector delta = Z.transform(sigmoidPrime).hadamard(deltaC);

      updateLayer(m_layers.size() - 1, delta, x, learnRate);

      // Back-propagate errors

      for (int i = m_layers.size() - 2; i >= 0; --i) {
        const Layer& nextLayer = m_layers[i + 1];
        Layer& thisLayer = m_layers[i];

        delta = nextLayer.weights.transposeMultiply(delta)
                                 .hadamard(thisLayer.Z.transform(sigmoidPrime));

        updateLayer(i, delta, x, learnRate);
      }

      learnRate *= learnRateDecay;
    }
  }
}

Vector NeuralNet::evaluate(const Vector& x) const {
  Vector A(x);
  std::cout << "Input: " << A;

  for (const auto& layer : m_layers) {
    A = (layer.weights * A + layer.biases).transform(sigmoid);
    std::cout << A;
  }

  std::cout << "Output: " << A;

  return A;
}
