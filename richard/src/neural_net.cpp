#include <cmath>
#include <iostream>
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

  return actual - expected;
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

Dataset::Dataset(const std::vector<char>& labels) : m_labels(labels) {
  for (size_t i = 0; i < m_labels.size(); ++i) {
    Vector v(m_labels.size());
    v.zero();
    v[i] = 1.0;
    m_classOutputVectors.insert({m_labels[i], v});
  }
}

void Dataset::normalize(const Vector& min, const Vector& max) {
  for (auto& sample : m_samples) {
    for (size_t i = 0; i < sample.data.size(); ++i) {
      sample.data[i] = (sample.data[i] - min[i]) / (max[i] - min[i]);
    }
  }
}

TrainingData::TrainingData(std::unique_ptr<Dataset> data)
  : m_data(std::move(data))
  , m_min(1)
  , m_max(1) {}

void TrainingData::normalize() {
  ASSERT(!m_data->samples().empty());

  auto& samples = m_data->samples();

  m_min = Vector(samples[0].data.size());
  m_max = Vector(samples[0].data.size());

  m_min.fill(std::numeric_limits<double>::max());
  m_max.fill(std::numeric_limits<double>::min());

  for (auto& sample : samples) {
    //sample.data.normalize();

    for (size_t i = 0; i < sample.data.size(); ++i) {
      if (sample.data[i] < m_min[i]) {
        m_min[i] = sample.data[i];
      }
      if (sample.data[i] > m_max[i]) {
        m_max[i] = sample.data[i];
      }
    }
  }

  m_data->normalize(m_min, m_max);
}

TestData::TestData(std::unique_ptr<Dataset> data)
  : m_data(std::move(data)) {}

void TestData::normalize(const Vector& trainingMin, const Vector& trainingMax) {
  ASSERT(!m_data->samples().empty());

  Vector min = trainingMin;
  Vector max = trainingMax;

  auto& samples = m_data->samples();

  for (auto& sample : samples) {
    for (size_t i = 0; i < sample.data.size(); ++i) {
      if (sample.data[i] < min[i]) {
        min[i] = sample.data[i];
      }
      if (sample.data[i] > max[i]) {
        max[i] = sample.data[i];
      }
    }
  }

  m_data->normalize(min, max);
}

NeuralNet::Layer::Layer(Layer&& mv)
  : weights(std::move(mv.weights))
  , biases(std::move(mv.biases))
  , Z(std::move(mv.Z))
  , A(std::move(mv.A)) {}

NeuralNet::Layer::Layer(Matrix&& weights, Vector&& biases)
  : weights(std::move(weights))
  , biases(std::move(biases))
  , Z(1)
  , A(1) {}

NeuralNet::Layer::Layer(const Layer& cpy)
  : weights(cpy.weights)
  , biases(cpy.biases)
  , Z(cpy.Z)
  , A(cpy.A) {}

NeuralNet::Layer::Layer(const Matrix& weights, const Vector& biases)
  : weights(weights)
  , biases(biases)
  , Z(1)
  , A(1) {}

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

void NeuralNet::toFile(const TrainingData& trainingData, const std::string& filePath) const {
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

  const Vector& min = trainingData.min();
  const Vector& max = trainingData.max();

  ASSERT(min.size() == m_numInputs);
  ASSERT(max.size() == m_numInputs);

  fout.write(reinterpret_cast<const char*>(min.data()), m_numInputs * sizeof(double));
  fout.write(reinterpret_cast<const char*>(max.data()), m_numInputs * sizeof(double));
}

void NeuralNet::fromFile(const std::string& filePath, Vector& trainingDataMin,
  Vector& trainingDataMax) {
  
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

  Vector min(m_numInputs);
  Vector max(m_numInputs);
  fin.read(reinterpret_cast<char*>(min.data()), m_numInputs * sizeof(double));
  fin.read(reinterpret_cast<char*>(max.data()), m_numInputs * sizeof(double));

  trainingDataMin = min;
  trainingDataMax = max;
}

size_t NeuralNet::inputSize() const {
  return m_numInputs;
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

void NeuralNet::updateLayer(size_t layerIdx, const Vector& delta, const Vector& x,
  double learnRate) {

  Layer& layer = m_layers[layerIdx];
  Vector prevLayerActivations = layerIdx == 0 ? x : m_layers[layerIdx - 1].A;

  for (size_t j = 0; j < layer.weights.rows(); j++) {
    for (size_t k = 0; k < layer.weights.cols(); k++) {
      double dw = prevLayerActivations[k] * delta[j] * learnRate;
      double w = layer.weights.at(k, j);
      layer.weights.set(k, j, w - dw);
    }
  }

  layer.biases = layer.biases - delta * learnRate;
}

double NeuralNet::feedForward(const Vector& x, const Vector& y, double dropoutRate) {
  auto shouldDrop = [dropoutRate]() {
    return rand() / (RAND_MAX + 1.0) < dropoutRate;
  };

  const Vector* A = nullptr;
  size_t i = 0;
  for (Layer& layer : m_layers) {
    layer.Z = layer.weights * (i == 0 ? x : *A) + layer.biases;
    layer.A = layer.Z.transform(sigmoid);
    A = &layer.A;

    if (i + 1 != m_layers.size()) {
      for (size_t a = 0; a < layer.A.size(); ++a) {
        if (shouldDrop()) {
          layer.A[a] = 0.0;
        }
      }
    }

    ++i;
  }

  return quadradicCost(*A, y);
}

void NeuralNet::train(const TrainingData& trainingData) {
  const Dataset& data = trainingData.data();
  const std::vector<Sample>& samples = data.samples();
  const size_t epochs = 50;
  double learnRate = 0.7;
  const double learnRateDecay = 1.0;
  const size_t maxSamplesToProcess = 10000;
  const size_t samplesToProcess = std::min<size_t>(maxSamplesToProcess, samples.size());
  const double dropoutRate = 0.5;

  std::cout << "Epochs: " << epochs << std::endl;
  std::cout << "Initial learn rate: " << learnRate << std::endl;
  std::cout << "Learn rate decay: " << learnRateDecay << std::endl;
  std::cout << "Samples in batch: " << samplesToProcess << std::endl;
  std::cout << "Dropout rate: " << dropoutRate << std::endl;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << "/" << epochs;

    double cost = 0.0;

    for (size_t i = 0; i < samplesToProcess; ++i) {
      const auto& sample = samples[i];
      const Vector& x = sample.data;
      const Vector& y = data.classOutputVector(sample.label);

      cost += feedForward(x, y, dropoutRate);

      Layer& outputLayer = m_layers.back();
      const Vector& Z = outputLayer.Z;

      Vector deltaC = quadraticCostDerivatives(outputLayer.A, y);
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
    }

    learnRate *= learnRateDecay;

    cost = cost / samplesToProcess;
    std::cout << ", cost = " << cost << std::endl;
  }
}

NeuralNet::Results NeuralNet::test(const TestData& testData) const {
  Results results;

  const Dataset& data = testData.data();
  const auto& samples = data.samples();

  double totalCost = 0.0;
  for (const auto& sample : samples) {
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

  results.cost = totalCost / data.samples().size();

  return results;
}

Vector NeuralNet::evaluate(const Vector& x) const {
  Vector A(x);

  for (const auto& layer : m_layers) {
    A = (layer.weights * A + layer.biases).transform(sigmoid);
  }

  return A;
}
