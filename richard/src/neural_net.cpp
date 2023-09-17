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

  Vector diff = expected - actual;
  return diff.hadamard(diff).sum() / 2.0;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
                                                      const Vector& expected) -> Vector {
  ASSERT(actual.size() == expected.size());

  return actual - expected;
};

struct Layer {
  virtual Matrix& W() = 0;
  virtual const Matrix& W() const = 0;
  virtual Vector& B() = 0;
  virtual const Vector& B() const = 0;
  virtual Vector& Z() = 0;
  virtual const Vector& Z() const = 0;
  virtual Vector& A() = 0;
  virtual const Vector& A() const = 0;

  virtual nlohmann::json getConfig() const = 0;

  virtual ~Layer() {}
};

class DenseLayer : public Layer {
  public:
    DenseLayer(DenseLayer&& mv);
    DenseLayer(Matrix&& weights, Vector&& biases);
    DenseLayer(const DenseLayer& cpy);
    DenseLayer(const Matrix& weights, const Vector& biases);

    Matrix& W() override { return m_weights; }
    const Matrix& W() const override { return m_weights; }
    Vector& B() override { return m_biases; }
    const Vector& B() const override { return m_biases; }
    Vector& Z() override { return m_Z; };
    const Vector& Z() const override { return m_Z; };
    Vector& A() override { return m_A; }
    const Vector& A() const override { return m_A; }

    nlohmann::json getConfig() const override;

  private:
    Matrix m_weights;
    Vector m_biases;
    Vector m_Z;
    Vector m_A;
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

    // For unit tests
    void setWeights(const std::vector<Matrix>& W) override;
    void setBiases(const std::vector<Vector>& B) override;

  private:
    double feedForward(const Vector& x, const Vector& y, double dropoutRate);
    void updateLayer(size_t layerIdx, const Vector& delta, const Vector& x, double learnRate);
    nlohmann::json getConfig() const;

    Hyperparams m_params;
    std::vector<std::unique_ptr<Layer>> m_layers;
    bool m_isTrained;
};

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

DenseLayer::DenseLayer(DenseLayer&& mv)
  : m_weights(std::move(mv.m_weights))
  , m_biases(std::move(mv.m_biases))
  , m_Z(std::move(mv.m_Z))
  , m_A(std::move(mv.m_A)) {}

DenseLayer::DenseLayer(Matrix&& weights, Vector&& biases)
  : m_weights(std::move(weights))
  , m_biases(std::move(biases))
  , m_Z(1)
  , m_A(1) {}

DenseLayer::DenseLayer(const DenseLayer& cpy)
  : m_weights(cpy.m_weights)
  , m_biases(cpy.m_biases)
  , m_Z(cpy.m_Z)
  , m_A(cpy.m_A) {}

DenseLayer::DenseLayer(const Matrix& weights, const Vector& biases)
  : m_weights(weights)
  , m_biases(biases)
  , m_Z(1)
  , m_A(1) {}

nlohmann::json DenseLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "dense";
  config["size"] = m_biases.size();
  return config;
}

std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, std::istream& fin,
  size_t prevLayerSize) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();

  Vector B(numNeurons);
  fin.read(reinterpret_cast<char*>(B.data()), numNeurons * sizeof(double));

  Matrix W(prevLayerSize, numNeurons);
  fin.read(reinterpret_cast<char*>(W.data()), W.rows() * W.cols() * sizeof(double));

  return std::make_unique<DenseLayer>(std::move(W), std::move(B));
}

std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, size_t prevLayerSize) {
  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();

  Vector B(numNeurons);
  B.randomize(1.0);

  Matrix W(prevLayerSize, numNeurons);
  W.randomize(1.0);

  return std::make_unique<DenseLayer>(std::move(W), std::move(B));
}

NeuralNetImpl::NeuralNetImpl(const nlohmann::json& config)
  : m_isTrained(false)
  , m_params(getOrThrow(config, "hyperparams")) {

  size_t prevLayerSize = m_params.numInputs;
  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize));
      prevLayerSize = m_layers.back()->B().size();
    }
  }

  nlohmann::json outputLayerJson;
  outputLayerJson["type"] = "dense";
  outputLayerJson["size"] = m_params.numOutputs;
  m_layers.push_back(constructLayer(outputLayerJson, prevLayerSize));
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
    prevLayerSize = m_layers.back()->B().size();
  }
  nlohmann::json outputLayerJson;
  outputLayerJson["type"] = "dense";
  outputLayerJson["size"] = m_params.numOutputs;
  m_layers.push_back(constructLayer(outputLayerJson, fin, prevLayerSize));

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
    const Layer& layer = *pLayer;
    const auto& B = layer.B();
    const auto& W = layer.W();

    fout.write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(double));
    fout.write(reinterpret_cast<const char*>(W.data()), W.rows() * W.cols() * sizeof(double));
  }
}

size_t NeuralNetImpl::inputSize() const {
  return m_params.numInputs;
}

void NeuralNetImpl::setWeights(const std::vector<Matrix>& W) {
  if (W.size() != m_layers.size()) {
    EXCEPTION("Wrong number of weight matrices");
  }

  for (size_t i = 0; i < W.size(); ++i) {
    m_layers[i]->W() = W[i];
  }
}

void NeuralNetImpl::setBiases(const std::vector<Vector>& B) {
  if (B.size() != m_layers.size()) {
    EXCEPTION("Wrong number of bias vectors");
  }

  for (size_t i = 0; i < B.size(); ++i) {
    m_layers[i]->B() = B[i];
  }
}

void NeuralNetImpl::updateLayer(size_t layerIdx, const Vector& delta, const Vector& x,
  double learnRate) {

  Layer& layer = *m_layers[layerIdx];
  Vector prevLayerActivations = layerIdx == 0 ? x : m_layers[layerIdx - 1]->A();

  for (size_t j = 0; j < layer.W().rows(); j++) {
    for (size_t k = 0; k < layer.W().cols(); k++) {
      double dw = prevLayerActivations[k] * delta[j] * learnRate;
      double w = layer.W().at(k, j);
      layer.W().set(k, j, w - dw);
    }
  }

  layer.B() = layer.B() - delta * learnRate;
}

double NeuralNetImpl::feedForward(const Vector& x, const Vector& y, double dropoutRate) {
  auto shouldDrop = [dropoutRate]() {
    return rand() / (RAND_MAX + 1.0) < dropoutRate;
  };

  const Vector* A = nullptr;
  size_t i = 0;
  for (auto& pLayer : m_layers) {
    Layer& layer = *pLayer;
    layer.Z() = layer.W() * (i == 0 ? x : *A) + layer.B();
    layer.A() = layer.Z().transform(sigmoid);
    A = &layer.A();

    if (i + 1 != m_layers.size()) {
      for (size_t a = 0; a < layer.A().size(); ++a) {
        if (shouldDrop()) {
          layer.A()[a] = 0.0;
        }
      }
    }

    ++i;
  }

  return quadradicCost(*A, y);
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

        Layer& outputLayer = *m_layers.back();
        const Vector& Z = outputLayer.Z();

        Vector deltaC = quadraticCostDerivatives(outputLayer.A(), y);
        Vector delta = Z.transform(sigmoidPrime).hadamard(deltaC);

        updateLayer(m_layers.size() - 1, delta, x, learnRate);

        // Back-propagate errors

        for (int l = m_layers.size() - 2; l >= 0; --l) {
          const Layer& nextLayer = *m_layers[l + 1];
          Layer& thisLayer = *m_layers[l];

          delta = nextLayer.W().transposeMultiply(delta)
                               .hadamard(thisLayer.Z().transform(sigmoidPrime));

          updateLayer(l, delta, x, learnRate);
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
  Vector A(x);

  for (const auto& layer : m_layers) {
    A = (layer->W() * A + layer->B()).transform(sigmoid);
  }

  return A;
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
