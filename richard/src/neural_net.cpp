#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <atomic>
#include "neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"
#include "dense_layer.hpp"
#include "max_pooling_layer.hpp"
#include "convolutional_layer.hpp"
#include "output_layer.hpp"

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  size_t n = expected.size();
  ASSERT(actual.size() == n);

  return (expected - actual).squareMagnitude() / static_cast<double>(n);
};

namespace {

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const nlohmann::json& obj);

  std::array<size_t, 2> numInputs;
  size_t epochs;
  size_t maxBatchSize;

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
    VectorPtr evaluate(const Array3& inputs) const override;

    void abort() override;

    // Exposed for testing
    //
    void setWeights(const std::vector<std::vector<DataArray>>& weights) override;
    void setBiases(const std::vector<DataArray>& biases) override;

  private:
    double feedForward(const Array3& x, const Vector& y);
    nlohmann::json getConfig() const;
    OutputLayer& outputLayer();
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, std::istream& fin,
      const std::array<size_t, 3>& prevLayerSize);
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj,
      const std::array<size_t, 3>& prevLayerSize);

    bool m_isTrained;
    Hyperparams m_params;
    std::vector<std::unique_ptr<Layer>> m_layers;
    std::atomic<bool> m_abort;
};

Hyperparams::Hyperparams()
  : numInputs({784, 1})
  , epochs(50)
  , maxBatchSize(1000) {}

Hyperparams::Hyperparams(const nlohmann::json& obj) {
  nlohmann::json params = Hyperparams().toJson();
  params.merge_patch(obj);
  numInputs = getOrThrow(params, "numInputs").get<std::array<size_t, 2>>();
  epochs = getOrThrow(params, "epochs").get<size_t>();
  maxBatchSize = getOrThrow(params, "maxBatchSize").get<size_t>();
}

nlohmann::json Hyperparams::toJson() const {
  nlohmann::json obj;

  obj["numInputs"] = numInputs;
  obj["epochs"] = epochs;
  obj["maxBatchSize"] = maxBatchSize;

  return obj;
}

void NeuralNetImpl::abort() {
  m_abort = true;
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj, std::istream& fin,
  const std::array<size_t, 3>& prevLayerSize) {

  size_t numInputs = prevLayerSize[0] * prevLayerSize[1] * prevLayerSize[2];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, fin, numInputs);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, fin, prevLayerSize[0], prevLayerSize[1],
      prevLayerSize[2]);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize[0], prevLayerSize[1],
      prevLayerSize[2]);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj,
  const std::array<size_t, 3>& prevLayerSize) {

  size_t numInputs = prevLayerSize[0] * prevLayerSize[1] * prevLayerSize[2];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, numInputs);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, prevLayerSize[0], prevLayerSize[1],
      prevLayerSize[2]);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize[0], prevLayerSize[1],
      prevLayerSize[2]);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

NeuralNetImpl::NeuralNetImpl(const nlohmann::json& config)
  : m_isTrained(false)
  , m_params(getOrThrow(config, "hyperparams")) {

  std::array<size_t, 3> prevLayerSize = { m_params.numInputs[0], m_params.numInputs[1], 1 };
  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  auto outLayerJson = config["outputLayer"];
  m_layers.push_back(std::make_unique<OutputLayer>(outLayerJson,
    prevLayerSize[0] * prevLayerSize[1] * prevLayerSize[2]));
}

NeuralNetImpl::NeuralNetImpl(std::istream& fin) : m_isTrained(false) {
  size_t configSize = 0;
  fin.read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  fin.read(reinterpret_cast<char*>(configString.data()), configSize);
  nlohmann::json config = nlohmann::json::parse(configString);

  nlohmann::json paramsJson = getOrThrow(config, "hyperparams");
  nlohmann::json layersJson = getOrThrow(config, "hiddenLayers");
  nlohmann::json outLayerJson = getOrThrow(config, "outputLayer");

  m_params = Hyperparams(paramsJson);

  std::array<size_t, 3> prevLayerSize = { m_params.numInputs[0], m_params.numInputs[1], 1 };
  for (auto& layerJson : layersJson) {
    m_layers.push_back(constructLayer(layerJson, fin, prevLayerSize));
    prevLayerSize = m_layers.back()->outputSize();
  }
  m_layers.push_back(std::make_unique<OutputLayer>(outLayerJson, fin,
    prevLayerSize[0] * prevLayerSize[1] * prevLayerSize[2]));

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
  layerJsons.pop_back(); // Output layer
  config["hiddenLayers"] = layerJsons;
  config["outputLayer"] = m_layers.back()->getConfig();
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

double NeuralNetImpl::feedForward(const Array3& x, const Vector& y) {
  const DataArray* A = &x.storage();
  for (auto& layer : m_layers) {
    //std::cout << "In: " << *A << std::endl;
  
    layer->trainForward(*A);
    A = &layer->activations();
    
    //std::cout << "Out: " << *A << std::endl;
  }

  ConstVectorPtr outputs = Vector::createShallow(*A);

  return quadradicCost(*outputs, y);
}

OutputLayer& NeuralNetImpl::outputLayer() {
  TRUE_OR_THROW(!m_layers.empty(), "No output layer");
  return dynamic_cast<OutputLayer&>(*m_layers.back());
}

void NeuralNetImpl::train(LabelledDataSet& trainingData) {
  std::cout << "Epochs: " << m_params.epochs << std::endl;
  std::cout << "Max batch size: " << m_params.maxBatchSize << std::endl;

  const size_t N = 500; // TODO

  m_abort = false;
  for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    if (m_abort) {
      break;
    }

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
        const Array3& x = sample.data;
        const Vector& y = trainingData.classOutputVector(sample.label);

        cost += feedForward(x, y);

        for (int l = static_cast<int>(m_layers.size()) - 1; l >= 0; --l) {
          if (l == static_cast<int>(m_layers.size()) - 1) {
            outputLayer().updateDelta(m_layers[m_layers.size() - 2]->activations(), y.storage(),
              epoch);
          }
          else if (l == 0) {
            m_layers[l]->updateDelta(x.storage(), *m_layers[l + 1], epoch);
          }
          else {
            m_layers[l]->updateDelta(m_layers[l - 1]->activations(), *m_layers[l + 1], epoch);
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

    cost = cost / samplesProcessed;
    std::cout << ", cost = " << cost << std::endl;

    if (std::isnan(cost)) {
      exit(1); // TODO
    }

    trainingData.seekToBeginning();
  }

  m_isTrained = true;
}

VectorPtr NeuralNetImpl::evaluate(const Array3& x) const {
  DataArray A;

  for (size_t i = 0; i < m_layers.size(); ++i) {
    A = m_layers[i]->evalForward(i == 0 ? x.storage() : A);
  }

  return std::make_unique<Vector>(A);
}

}

const nlohmann::json& NeuralNet::defaultConfig() {
  static nlohmann::json config;
  static bool done = false;

  if (!done) {
    // TODO: Construct temporary layers and get config?

    nlohmann::json layer1;
    layer1["type"] = "dense";
    layer1["size"] = 300;
    layer1["learnRate"] = 0.7;
    layer1["learnRateDecay"] = 1.0;
    layer1["dropoutRate"] = 0.5;
    nlohmann::json layer2;
    layer2["type"] = "dense";
    layer2["size"] = 80;
    layer2["learnRate"] = 0.7;
    layer2["learnRateDecay"] = 1.0;
    layer2["dropoutRate"] = 0.5;
    std::vector<nlohmann::json> layersJson{layer1, layer2};

    config["hyperparams"] = Hyperparams().toJson();
    config["hiddenLayers"] = layersJson;

    nlohmann::json outLayer;
    outLayer["type"] = "output";
    outLayer["size"] = 10;
    outLayer["learnRate"] = 0.7;
    outLayer["learnRateDecay"] = 1.0;

    config["outputLayer"] = outLayer;

    done = true;
  }

  return config;
}

void NeuralNetImpl::setWeights(const std::vector<std::vector<DataArray>>& weights) {
  assert(m_layers.size() == weights.size());
  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (m_layers[i]->type() == LayerType::MAX_POOLING) {
      continue;
    }

    m_layers[i]->setWeights(weights[i]);
  }
}

void NeuralNetImpl::setBiases(const std::vector<DataArray>& biases) {
  assert(m_layers.size() == biases.size());
  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (m_layers[i]->type() == LayerType::MAX_POOLING) {
      continue;
    }

    m_layers[i]->setBiases(biases[i]);
  }
}

NeuralNetPtr createNeuralNet(const nlohmann::json& config) {
  return std::make_unique<NeuralNetImpl>(config);
}

NeuralNetPtr createNeuralNet(std::istream& fin) {
  return std::make_unique<NeuralNetImpl>(fin);
}

// TODO: Move this?
std::ostream& operator<<(std::ostream& os, LayerType layerType) {
  switch (layerType) {
    case LayerType::DENSE: os << "DENSE"; break;
    case LayerType::CONVOLUTIONAL: os << "CONVOLUTIONAL"; break;
    case LayerType::OUTPUT: os << "OUTPUT"; break;
    case LayerType::MAX_POOLING: os << "MAX_POOLING"; break;
  }
  return os;
}

