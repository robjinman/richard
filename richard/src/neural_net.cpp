#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"
#include "dense_layer.hpp"
#include "max_pooling_layer.hpp"
#include "convolutional_layer.hpp"
#include "output_layer.hpp"

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  ASSERT(actual.size() == expected.size());

  return (expected - actual).squareMagnitude() / 2.0;
};

namespace {

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const nlohmann::json& obj);

  std::array<size_t, 2> numInputs;
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
    std::array<size_t, 2> inputSize() const override;
    void writeToStream(std::ostream& s) const override;
    void train(LabelledDataSet& data) override;
    Vector evaluate(const Vector& inputs) const override;

  private:
    double feedForward(const Vector& x, const Vector& y, double dropoutRate);
    nlohmann::json getConfig() const;
    OutputLayer& outputLayer();
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, std::istream& fin,
      const std::array<size_t, 2>& prevLayerSize);
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj,
      const std::array<size_t, 2>& prevLayerSize);

    bool m_isTrained;
    Hyperparams m_params;
    std::vector<std::unique_ptr<Layer>> m_layers;
};

Hyperparams::Hyperparams()
  : numInputs({784, 1})
  , numOutputs(10)
  , epochs(50)
  , learnRate(0.7)
  , learnRateDecay(1.0)
  , maxBatchSize(1000)
  , dropoutRate(0.5) {}

Hyperparams::Hyperparams(const nlohmann::json& obj) {
  nlohmann::json params = Hyperparams().toJson();
  params.merge_patch(obj);
  numInputs = getOrThrow(params, "numInputs").get<std::array<size_t, 2>>();
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
  const std::array<size_t, 2>& prevLayerSize) {

  size_t numInputs = prevLayerSize[0] * prevLayerSize[1];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, fin, numInputs, m_params.learnRate,
      m_params.dropoutRate);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, fin, prevLayerSize[0], prevLayerSize[1],
      m_params.learnRate);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize[0], prevLayerSize[1]);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj,
  const std::array<size_t, 2>& prevLayerSize) {

  size_t numInputs = prevLayerSize[0] * prevLayerSize[1];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, numInputs, m_params.learnRate,
      m_params.dropoutRate);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, prevLayerSize[0], prevLayerSize[1],
      m_params.learnRate);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize[0], prevLayerSize[1]);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

NeuralNetImpl::NeuralNetImpl(const nlohmann::json& config)
  : m_isTrained(false)
  , m_params(getOrThrow(config, "hyperparams")) {

  auto prevLayerSize = m_params.numInputs;
  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  m_layers.push_back(std::make_unique<OutputLayer>(m_params.numOutputs,
    prevLayerSize[0] * prevLayerSize[1], m_params.learnRate));
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

  auto prevLayerSize = m_params.numInputs;
  for (auto& layerJson : layersJson) {
    m_layers.push_back(constructLayer(layerJson, fin, prevLayerSize));
    prevLayerSize = m_layers.back()->outputSize();
  }
  m_layers.push_back(std::make_unique<OutputLayer>(fin, m_params.numOutputs,
    prevLayerSize[0] * prevLayerSize[1], m_params.learnRate));

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

std::array<size_t, 2> NeuralNetImpl::inputSize() const {
  return m_params.numInputs;
}

std::ostream& operator<<(std::ostream& os, LayerType layerType) {
  switch (layerType) {
    case LayerType::DENSE: os << "DENSE"; break;
    case LayerType::CONVOLUTIONAL: os << "CONVOLUTIONAL"; break;
    case LayerType::OUTPUT: os << "OUTPUT"; break;
    case LayerType::MAX_POOLING: os << "MAX_POOLING"; break;
  }
  return os;
}

double NeuralNetImpl::feedForward(const Vector& x, const Vector& y, double dropoutRate) {
  const Vector* A = &x;
  for (auto& layer : m_layers) {
    //std::cout << "Layer type: " << layer->type() << "\n";
    //std::cout << "In: \n";
    //std::cout << *A;
    layer->trainForward(*A);
    //std::cout << "Out: \n";
    A = &layer->activations();
    //std::cout << *A;
  }

  return quadradicCost(*A, y);
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
      size_t netInputSize = m_params.numInputs[0] * m_params.numInputs[1];
      TRUE_OR_THROW(samples[0].data.size() == netInputSize,
        "Sample size is " << samples[0].data.size() << ", expected " << netInputSize);

      for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const Vector& x = sample.data;
        const Vector& y = trainingData.classOutputVector(sample.label);

        cost += feedForward(x, y, m_params.dropoutRate);

        for (int l = static_cast<int>(m_layers.size()) - 1; l >= 0; --l) {
          if (l == static_cast<int>(m_layers.size()) - 1) {
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
    std::cout << ", learn rate = " << learnRate << ", cost = " << cost << std::endl;

    if (std::isnan(cost)) {
      exit(1); // TODO
    }

    trainingData.seekToBeginning();
  }

  m_isTrained = true;
}

Vector NeuralNetImpl::evaluate(const Vector& x) const {
  Vector A = x;
  for (const auto& layer : m_layers) {
    A = layer->evalForward(A);
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
