#include <cmath>
#include <iostream> // TODO
#include <fstream>
#include <algorithm>
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

bool outputsMatch(const Vector& x, const Vector& y) {
  auto largestComponent = [](const Vector& v) {
    double largest = std::numeric_limits<double>::min();
    size_t largestIdx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i] > largest) {
        largest = v[i];
        largestIdx = i;
      }
    }
    return largestIdx;
  };

  return largestComponent(x) == largestComponent(y);
}

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

NeuralNet::Layer::Layer(Layer&& mv)
  : weights(std::move(mv.weights))
  , biases(std::move(mv.biases))
  , Z(std::move(mv.Z)) {}

NeuralNet::Layer::Layer(Matrix&& weights, Vector&& biases)
  : weights(std::move(weights))
  , biases(std::move(biases))
  , Z(1) {}

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
      m_numInputs = layerSize;
      prevLayerSize = layerSize;
      ++i;
      continue;
    }

    Matrix weights(prevLayerSize, layerSize);
    weights.randomize(1.0);

    Vector biases(layerSize);
    biases.randomize(1.0);

    m_layers.emplace_back(std::move(weights), std::move(biases));

    prevLayerSize = layerSize;
    ++i;
  }
}

void NeuralNet::toFile(const std::string& filePath) const {
  std::ofstream fout(filePath, std::ios::out | std::ios::binary);

  size_t numLayers = m_layers.size();
  fout.write(reinterpret_cast<char*>(&numLayers), sizeof(size_t));

  fout.write(reinterpret_cast<const char*>(&m_numInputs), sizeof(size_t));

  for (const auto& layer : m_layers) {
    size_t numNeurons = layer.biases.size();
    const auto& B = layer.biases;
    const auto& W = layer.weights;

    fout.write(reinterpret_cast<char*>(&numNeurons), sizeof(size_t));
    fout.write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(double));
    fout.write(reinterpret_cast<const char*>(W.data()), W.rows() * W.cols() * sizeof(double));
  }
}

void NeuralNet::fromFile(const std::string& filePath) {
  std::ifstream fin(filePath, std::ios::in | std::ios::binary);

  m_layers.clear();
  m_numInputs = 0;

  size_t numLayers = 0;
  fin.read(reinterpret_cast<char*>(&numLayers), sizeof(size_t));

  fin.read(reinterpret_cast<char*>(&m_numInputs), sizeof(size_t));

  size_t prevLayerSize = m_numInputs;
  for (size_t i = 0; i < numLayers; ++i) {
    size_t numNeurons = 0;
    fin.read(reinterpret_cast<char*>(&numNeurons), sizeof(size_t));

    Vector B(numNeurons);
    fin.read(reinterpret_cast<char*>(B.data()), numNeurons * sizeof(double));

    Matrix W(prevLayerSize, numNeurons);
    fin.read(reinterpret_cast<char*>(W.data()), W.rows() * W.cols() * sizeof(double));

    m_layers.emplace_back(std::move(W), std::move(B));

    prevLayerSize = numNeurons;
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
  const size_t epochs = 100; // TODO
  const double initialLearnRate = 0.5;
  const double learnRateDecay = 1.0;
  const size_t maxSamplesToProcess = 300;
  const size_t samplesToProcess = std::min<size_t>(maxSamplesToProcess, samples.size()); // TODO

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl; // TODO

    double learnRate = initialLearnRate;

    for (size_t i = 0; i < samplesToProcess; ++i) {
      //std::cout << i << "/" << samplesToProcess << std::endl; // TODO

      const auto& sample = samples[i];
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

      for (int l = m_layers.size() - 2; l >= 0; --l) {
        const Layer& nextLayer = m_layers[l + 1];
        Layer& thisLayer = m_layers[l];

        delta = nextLayer.weights.transposeMultiply(delta)
                                 .hadamard(thisLayer.Z.transform(sigmoidPrime));

        updateLayer(l, delta, x, learnRate);
      }

      learnRate *= learnRateDecay;
    }
  }
}

NeuralNet::Results NeuralNet::test(const TrainingData& data) const {
  Results results;

  double totalCost = 0.0;
  for (const auto& sample : data.data()) {
    Vector actual = evaluate(sample.data);
    Vector expected = data.classOutputVector(sample.label);

    if (outputsMatch(actual, expected)) {
      ++results.good;
      std::cout << "1" << std::flush;
    }
    else {
      ++results.bad;
      std::cout << "0" << std::flush;
    }

    totalCost += quadradicCost(actual, expected);
  }
  std::cout << std::endl;

  results.cost = totalCost / data.data().size();

  return results;
}

Vector NeuralNet::evaluate(const Vector& x) const {
  Vector A(x);
  //std::cout << "Input: " << A;

  for (const auto& layer : m_layers) {
    A = (layer.weights * A + layer.biases).transform(sigmoid);
    //std::cout << A;
  }

  //std::cout << "Output: " << A;

  return A;
}
