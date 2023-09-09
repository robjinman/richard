#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"

namespace {

using ActivationFn = std::function<double(double)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](double x) -> double {
  return 1.0 / (1.0 + exp(-x));
};

const ActivationFn sigmoidPrime = [](double x) -> double {
  double sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
};

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
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

}

NetworkConfig::NetworkConfig(std::istream& s) {
  std::map<std::string, std::string> keyVals = readKeyValuePairs(s);

  std::stringstream ss(keyVals.at("layers"));
  std::string strLayer;
  while (std::getline(ss, strLayer, ',')) {
    layers.push_back(std::stoul(strLayer));
  }

  if (keyVals.count("epochs")) {
    params.epochs = std::stoul(keyVals.at("epochs"));
  }
  if (keyVals.count("learnRate")) {
    params.learnRate = std::stod(keyVals.at("learnRate"));
  }
  if (keyVals.count("learnRateDecay")) {
    params.learnRateDecay = std::stod(keyVals.at("learnRateDecay"));
  }
  if (keyVals.count("maxBatchSize")) {
    params.maxBatchSize = std::stoul(keyVals.at("maxBatchSize"));
  }
  if (keyVals.count("dropoutRate")) {
    params.dropoutRate = std::stod(keyVals.at("dropoutRate"));
  }
}

void NetworkConfig::writeToStream(std::ostream& s) const {
  s << "layers=";
  for (size_t i = 0; i < layers.size(); ++i) {
    s << layers[i];
    if (i + 1 < layers.size()) {
      s << ",";
    }
  }
  s << std::endl;
  s << "epochs=" << params.epochs << std::endl;
  s << "learnRate=" << params.learnRate << std::endl;
  s << "learnRateDecay=" << params.learnRateDecay << std::endl;
  s << "maxBatchSize=" << params.maxBatchSize << std::endl;
  s << "dropoutRate=" << params.dropoutRate << std::endl;
}

void NetworkConfig::printExample(std::ostream& s) {
  NetworkConfig config(std::vector<size_t>({784, 300, 80, 10}));
  config.writeToStream(s);
}

NetworkConfig::NetworkConfig(const std::vector<size_t>& layers)
  : layers(layers) {}

NetworkConfig NetworkConfig::fromFile(const std::string& filePath) {
  std::ifstream fin(filePath);
  return NetworkConfig(fin);
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

NeuralNet::Layer::Layer(const Matrix& weights, const Vector& biases)
  : weights(weights)
  , biases(biases)
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

NeuralNet::NeuralNet(const NetworkConfig& config)
  : m_config(config)
  , m_isTrained(false) {

  size_t prevLayerSize = 0;
  size_t i = 0;
  for (size_t layerSize : config.layers) {
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

NeuralNet::NeuralNet(std::istream& fin)
  : m_config(std::vector<size_t>())
  , m_isTrained(false) {

  size_t configSize = 0;
  fin.read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  fin.read(reinterpret_cast<char*>(configString.data()), configSize);

  std::stringstream ss(configString);
  m_config = NetworkConfig(ss);

  m_numInputs = m_config.layers[0];
  size_t numLayers = m_config.layers.size() - 1;

  size_t prevLayerSize = m_numInputs;
  for (size_t i = 0; i < numLayers; ++i) {
    size_t numNeurons = m_config.layers[i + 1];

    Vector B(numNeurons);
    fin.read(reinterpret_cast<char*>(B.data()), numNeurons * sizeof(double));

    Matrix W(prevLayerSize, numNeurons);
    fin.read(reinterpret_cast<char*>(W.data()), W.rows() * W.cols() * sizeof(double));

    m_layers.emplace_back(std::move(W), std::move(B));

    prevLayerSize = numNeurons;
  }

  m_isTrained = true;
}

NeuralNet::CostFn NeuralNet::costFn() const {
  return quadradicCost;
}

void NeuralNet::writeToStream(std::ostream& fout) const {
  TRUE_OR_THROW(m_isTrained, "Neural net is not trained");

  std::stringstream ss;
  m_config.writeToStream(ss);

  size_t configSize = ss.str().size();
  fout.write(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  fout.write(ss.str().c_str(), configSize);

  for (const auto& layer : m_layers) {
    const auto& B = layer.biases;
    const auto& W = layer.weights;

    fout.write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(double));
    fout.write(reinterpret_cast<const char*>(W.data()), W.rows() * W.cols() * sizeof(double));
  }
}

size_t NeuralNet::inputSize() const {
  return m_numInputs;
}

void NeuralNet::setWeights(const std::vector<Matrix>& W) {
  if (W.size() != m_layers.size()) {
    EXCEPTION("Wrong number of weight matrices");
  }

  for (size_t i = 0; i < W.size(); ++i) {
    m_layers[i].weights = W[i];
  }
}

void NeuralNet::setBiases(const std::vector<Vector>& B) {
  if (B.size() != m_layers.size()) {
    EXCEPTION("Wrong number of bias vectors");
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
  const HyperParams& params = m_config.params;
  const Dataset& data = trainingData.data();
  const std::vector<Sample>& samples = data.samples();
  double learnRate = params.learnRate;
  const size_t samplesToProcess = std::min<size_t>(params.maxBatchSize, samples.size());

  std::cout << "Epochs: " << params.epochs << std::endl;
  std::cout << "Initial learn rate: " << params.learnRate << std::endl;
  std::cout << "Learn rate decay: " << params.learnRateDecay << std::endl;
  std::cout << "Samples in batch: " << samplesToProcess << std::endl;
  std::cout << "Dropout rate: " << params.dropoutRate << std::endl;

  TRUE_OR_THROW(!samples.empty(), "Dataset is empty");
  TRUE_OR_THROW(samples[0].data.size() == m_numInputs,
    "Sample size is " << samples[0].data.size() << ", expected " << m_numInputs);

  for (size_t epoch = 0; epoch < params.epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << "/" << params.epochs;

    double cost = 0.0;

    for (size_t i = 0; i < samplesToProcess; ++i) {
      const auto& sample = samples[i];
      const Vector& x = sample.data;
      const Vector& y = data.classOutputVector(sample.label);

      cost += feedForward(x, y, params.dropoutRate);

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

    learnRate *= params.learnRateDecay;

    cost = cost / samplesToProcess;
    std::cout << ", cost = " << cost << std::endl;
  }

  m_isTrained = true;
}

Vector NeuralNet::evaluate(const Vector& x) const {
  Vector A(x);

  for (const auto& layer : m_layers) {
    A = (layer.weights * A + layer.biases).transform(sigmoid);
  }

  return A;
}
