#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"

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

  return (expected - actual).squareMagnitude() / 2.0;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
                                                      const Vector& expected) -> Vector {
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
    virtual size_t outputSize() const = 0;
    virtual const Vector& activations() const = 0;
    virtual const Vector& delta() const = 0;
    virtual void trainForward(const Vector& inputs) = 0;
    virtual Vector evalForward(const Vector& inputs) const = 0;
    virtual void updateDelta(const Vector& layerInputs, const Layer& nextLayer) = 0;
    virtual nlohmann::json getConfig() const = 0;
    virtual void writeToStream(std::ostream& fout) const = 0;
    virtual const Matrix& W() const = 0;

    virtual ~Layer() {}
};

class OutputLayer : public Layer {
  public:
    OutputLayer(size_t numNeurons, size_t inputSize, double learnRate);

    LayerType type() const override { return LayerType::OUTPUT; }
    size_t outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override { assert(false); }
    void updateDelta(const Vector& layerInputs, const Vector& y);
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const {}
    const Matrix& W() const override;

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    double m_learnRate;
};

class DenseLayer : public Layer {
  public:
    DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize, double learnRate,
      double dropoutRate);
    DenseLayer(const nlohmann::json& obj, size_t inputSize, double learnRate, double dropoutRate);

    LayerType type() const override { return LayerType::DENSE; }
    size_t outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const;
    const Matrix& W() const override;

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    double m_learnRate;
    double m_dropoutRate;
};

class ConvolutionalLayer : public Layer {
  public:
    //ConvolutionalLayer(size_t width, size_t height, const Matrix& weights, const Vector& biases);

    LayerType type() const override { return LayerType::CONVOLUTIONAL; }
    size_t outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const;
    const Matrix& W() const override;

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
};

class MaxPoolingLayer : public Layer {
  public:
    //MaxPoolingLayer(size_t regionW, size_t regionH);

    LayerType type() const override { return LayerType::MAX_POOLING; }
    size_t outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const;
    const Matrix& W() const override { assert(false); }

  private:
    Vector m_Z;
    Vector m_delta;
};

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const nlohmann::json& obj);

  size_t numInputs;
  size_t numOutputs;
  size_t epochs;
  double learnRate;
  double learnRateDecay;
  size_t maxBatchSize;
  double dropoutRate;

  nlohmann::json toJson() const;
};

class NeuralNetImpl : public NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    explicit NeuralNetImpl(const nlohmann::json& config);
    explicit NeuralNetImpl(std::istream& s);

    CostFn costFn() const override;
    size_t inputSize() const override;
    void writeToStream(std::ostream& s) const override;
    void train(LabelledDataSet& data) override;
    Vector evaluate(const Vector& inputs) const override;

  private:
    double feedForward(const Vector& x, const Vector& y, double dropoutRate);
    nlohmann::json getConfig() const;
    OutputLayer& outputLayer();
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, std::istream& fin,
      size_t prevLayerSize);
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, size_t prevLayerSize);

    Hyperparams m_params;
    std::vector<std::unique_ptr<Layer>> m_layers;
    bool m_isTrained;
};

OutputLayer::OutputLayer(size_t numNeurons, size_t inputSize, double learnRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate) {

  m_B = Vector(numNeurons);
  m_B.randomize(1.0);

  m_W = Matrix(inputSize, numNeurons);
  m_W.randomize(1.0);
}

const Vector& OutputLayer::activations() const {
  return m_A;
}

const Vector& OutputLayer::delta() const {
  return m_delta;
}

const Matrix& OutputLayer::W() const {
  return m_W;
}

nlohmann::json OutputLayer::getConfig() const {
  return nlohmann::json();
}

Vector OutputLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

size_t OutputLayer::outputSize() const {
  return m_B.size();
}

void OutputLayer::trainForward(const Vector& inputs) {
  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);
}

void OutputLayer::updateDelta(const Vector& layerInputs, const Vector& y) {
  Vector deltaC = quadraticCostDerivatives(m_A, y);
  m_delta = m_Z.transform(sigmoidPrime).hadamard(deltaC);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * m_learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * m_learnRate;
}

DenseLayer::DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize,
  double learnRate, double dropoutRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate)
  , m_dropoutRate(dropoutRate) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();

  m_B = Vector(numNeurons);
  fin.read(reinterpret_cast<char*>(m_B.data()), numNeurons * sizeof(double));

  m_W = Matrix(inputSize, numNeurons);
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

DenseLayer::DenseLayer(const nlohmann::json& obj, size_t inputSize, double learnRate,
  double dropoutRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate)
  , m_dropoutRate(dropoutRate) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();

  m_B = Vector(numNeurons);
  m_B.randomize(1.0);

  m_W = Matrix(inputSize, numNeurons);
  m_W.randomize(1.0);
}

void DenseLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

size_t DenseLayer::outputSize() const {
  return m_B.size();
}

const Vector& DenseLayer::activations() const {
  return m_A;
}

const Vector& DenseLayer::delta() const {
  return m_delta;
}

const Matrix& DenseLayer::W() const {
  return m_W;
}

nlohmann::json DenseLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "dense";
  config["size"] = m_B.size();
  return config;
}

Vector DenseLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

void DenseLayer::trainForward(const Vector& inputs) {
  auto shouldDrop = [this]() {
    return rand() / (RAND_MAX + 1.0) < m_dropoutRate;
  };

  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);

  for (size_t a = 0; a < m_A.size(); ++a) {
    if (shouldDrop()) {
      m_A[a] = 0.0;
    }
  }
}

void DenseLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {
  m_delta = nextLayer.W().transposeMultiply(nextLayer.delta())
                         .hadamard(m_Z.transform(sigmoidPrime));

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * m_learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * m_learnRate;
}

size_t ConvolutionalLayer::outputSize() const {
  return m_B.size();
}

void ConvolutionalLayer::trainForward(const Vector& inputs) {

}

Vector ConvolutionalLayer::evalForward(const Vector& inputs) const {

}

void ConvolutionalLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {

}

nlohmann::json ConvolutionalLayer::getConfig() const {

}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {

}

const Matrix& ConvolutionalLayer::W() const {

}

size_t MaxPoolingLayer::outputSize() const {

}

const Vector& MaxPoolingLayer::activations() const {

}

const Vector& MaxPoolingLayer::delta() const {

}

void MaxPoolingLayer::trainForward(const Vector& inputs) {

}

Vector MaxPoolingLayer::evalForward(const Vector& inputs) const {

}

void MaxPoolingLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {

}

void MaxPoolingLayer::writeToStream(std::ostream& fout) const {

}

nlohmann::json MaxPoolingLayer::getConfig() const {

}

Hyperparams::Hyperparams()
  : numInputs(784)
  , numOutputs(10)
  , epochs(50)
  , learnRate(0.7)
  , learnRateDecay(1.0)
  , maxBatchSize(1000)
  , dropoutRate(0.5) {}

Hyperparams::Hyperparams(const nlohmann::json& obj) {
  nlohmann::json params = Hyperparams().toJson();
  params.merge_patch(obj);
  numInputs = getOrThrow(params, "numInputs").get<size_t>();
  numOutputs = getOrThrow(params, "numOutputs").get<size_t>();
  epochs = getOrThrow(params, "epochs").get<size_t>();
  learnRate = getOrThrow(params, "learnRate").get<double>();
  learnRateDecay = getOrThrow(params, "learnRateDecay").get<double>();
  maxBatchSize = getOrThrow(params, "maxBatchSize").get<size_t>();
  dropoutRate = getOrThrow(params, "dropoutRate").get<double>();
}

nlohmann::json Hyperparams::toJson() const {
  nlohmann::json obj;

  obj["numInputs"] = numInputs;
  obj["numOutputs"] = numOutputs;
  obj["epochs"] = epochs;
  obj["learnRate"] = learnRate;
  obj["learnRateDecay"] = learnRateDecay;
  obj["maxBatchSize"] = maxBatchSize;
  obj["dropoutRate"] = dropoutRate;

  return obj;
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj, std::istream& fin,
  size_t prevLayerSize) {

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, fin, prevLayerSize, m_params.learnRate,
      m_params.dropoutRate);
  }
  else if (type == "convolutional") {
    //return std::make_unique<ConvolutionalLayer>(obj, fin, prevLayerSize);
  }
  else if (type == "maxPooling") {
    //return std::make_unique<MaxPoolingLayer>(obj, fin, prevLayerSize);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj,
  size_t prevLayerSize) {

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, prevLayerSize, m_params.learnRate,
      m_params.dropoutRate);
  }
  else if (type == "convolutional") {
    //return std::make_unique<ConvolutionalLayer>(obj, prevLayerSize);
  }
  else if (type == "maxPooling") {
    //return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

NeuralNetImpl::NeuralNetImpl(const nlohmann::json& config)
  : m_isTrained(false)
  , m_params(getOrThrow(config, "hyperparams")) {

  size_t prevLayerSize = m_params.numInputs;
  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  m_layers.push_back(std::make_unique<OutputLayer>(m_params.numOutputs, prevLayerSize,
    m_params.learnRate));
}

NeuralNetImpl::NeuralNetImpl(std::istream& fin) : m_isTrained(false) {
  size_t configSize = 0;
  fin.read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  fin.read(reinterpret_cast<char*>(configString.data()), configSize);
  nlohmann::json config = nlohmann::json::parse(configString);

  nlohmann::json paramsJson = getOrThrow(config, "hyperparams");
  nlohmann::json layersJson = getOrThrow(config, "hiddenLayers");

  m_params = Hyperparams(paramsJson);

  size_t prevLayerSize = m_params.numInputs;
  for (auto& layerJson : layersJson) {
    m_layers.push_back(constructLayer(layerJson, fin, prevLayerSize));
    prevLayerSize = m_layers.back()->outputSize();
  }
  m_layers.push_back(std::make_unique<OutputLayer>(m_params.numOutputs, prevLayerSize,
    m_params.learnRate));

  m_isTrained = true;
}

NeuralNet::CostFn NeuralNetImpl::costFn() const {
  return quadradicCost;
}

nlohmann::json NeuralNetImpl::getConfig() const {
  nlohmann::json config;
  config["hyperparams"] = m_params.toJson();
  std::vector<nlohmann::json> layerJsons;
  for (auto& pLayer : m_layers) {
    layerJsons.push_back(pLayer->getConfig());
  }
  layerJsons.pop_back(); // Omit the output layer
  config["hiddenLayers"] = layerJsons;
  return config;
}

void NeuralNetImpl::writeToStream(std::ostream& fout) const {
  TRUE_OR_THROW(m_isTrained, "Neural net is not trained");

  std::string configString = getConfig().dump();
  size_t configSize = configString.size();
  fout.write(reinterpret_cast<char*>(&configSize), sizeof(size_t));
  fout.write(configString.c_str(), configSize);

  for (const auto& pLayer : m_layers) {
    pLayer->writeToStream(fout);
  }
}

size_t NeuralNetImpl::inputSize() const {
  return m_params.numInputs;
}

double NeuralNetImpl::feedForward(const Vector& x, const Vector& y, double dropoutRate) {
  const Vector* inputs = &x;
  for (auto& layer : m_layers) {
    layer->trainForward(*inputs);
    inputs = &layer->activations();
  }

  return quadradicCost(*inputs, y);
}

OutputLayer& NeuralNetImpl::outputLayer() {
  TRUE_OR_THROW(!m_layers.empty(), "No output layer");
  return dynamic_cast<OutputLayer&>(*m_layers.back());
}

void NeuralNetImpl::train(LabelledDataSet& trainingData) {
  double learnRate = m_params.learnRate;

  std::cout << "Epochs: " << m_params.epochs << std::endl;
  std::cout << "Initial learn rate: " << m_params.learnRate << std::endl;
  std::cout << "Learn rate decay: " << m_params.learnRateDecay << std::endl;
  std::cout << "Max batch size: " << m_params.maxBatchSize << std::endl;
  std::cout << "Dropout rate: " << m_params.dropoutRate << std::endl;

  const size_t N = 500; // TODO

  for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << "/" << m_params.epochs;
    double cost = 0.0;
    size_t samplesProcessed = 0;

    std::vector<Sample> samples;
    while (trainingData.loadSamples(samples, N) > 0) {
      TRUE_OR_THROW(samples[0].data.size() == m_params.numInputs,
        "Sample size is " << samples[0].data.size() << ", expected " << m_params.numInputs);

      for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const Vector& x = sample.data;
        const Vector& y = trainingData.classOutputVector(sample.label);

        cost += feedForward(x, y, m_params.dropoutRate);

        for (int l = m_layers.size() - 1; l >= 0; --l) {
          if (l == m_layers.size() - 1) {
            outputLayer().updateDelta(m_layers[m_layers.size() - 2]->activations(), y);
          }
          else if (l == 0) {
            m_layers[l]->updateDelta(x, *m_layers[l + 1]);
          }
          else {
            m_layers[l]->updateDelta(m_layers[l - 1]->activations(), *m_layers[l + 1]);
          }
        }

        ++samplesProcessed;
        if (samplesProcessed >= m_params.maxBatchSize) {
          break;
        }
      }

      samples.clear();

      if (samplesProcessed >= m_params.maxBatchSize) {
        break;
      }
    }

    learnRate *= m_params.learnRateDecay;

    cost = cost / samplesProcessed;
    std::cout << ", cost = " << cost << std::endl;

    trainingData.seekToBeginning();
  }

  m_isTrained = true;
}

Vector NeuralNetImpl::evaluate(const Vector& x) const {
  Vector activations = x;
  for (const auto& layer : m_layers) {
    activations = layer->evalForward(activations);
  }

  return activations;
}

}

const nlohmann::json& NeuralNet::defaultConfig() {
  static nlohmann::json config;
  static bool done = false;

  if (!done) {
    nlohmann::json layer1;
    layer1["type"] = "dense";
    layer1["size"] = 300;
    nlohmann::json layer2;
    layer2["type"] = "dense";
    layer2["size"] = 80;
    std::vector<nlohmann::json> layersJson{layer1, layer2};

    config["hyperparams"] = Hyperparams().toJson();
    config["hiddenLayers"] = layersJson;

    done = true;
  }

  return config;
}

std::unique_ptr<NeuralNet> createNeuralNet(const nlohmann::json& config) {
  return std::make_unique<NeuralNetImpl>(config);
}

std::unique_ptr<NeuralNet> createNeuralNet(std::istream& fin) {
  return std::make_unique<NeuralNetImpl>(fin);
}
