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
#include "logger.hpp"

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  DBG_ASSERT(actual.size() == expected.size());
  return (expected - actual).squareMagnitude() * 0.5;
};

namespace {

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const nlohmann::json& obj);

  size_t epochs;
  size_t maxBatchSize;

  static const nlohmann::json& exampleConfig();
};

class NeuralNetImpl : public NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    NeuralNetImpl(const Triple& inputShape, const nlohmann::json& config, Logger& logger);
    NeuralNetImpl(const Triple& inputShape, const nlohmann::json& config, std::istream& s,
      Logger& logger);

    CostFn costFn() const override;
    Triple inputSize() const override;
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
    OutputLayer& outputLayer();
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, std::istream& fin,
      const Triple& prevLayerSize);
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, const Triple& prevLayerSize);

    Logger& m_logger;
    bool m_isTrained;
    Triple m_inputShape;
    Hyperparams m_params;
    std::vector<std::unique_ptr<Layer>> m_layers;
    std::atomic<bool> m_abort;
};

Hyperparams::Hyperparams()
  : epochs(0)
  , maxBatchSize(1000) {}

Hyperparams::Hyperparams(const nlohmann::json& obj) {
  epochs = getOrThrow(obj, "epochs").get<size_t>();
  maxBatchSize = getOrThrow(obj, "maxBatchSize").get<size_t>();
}

const nlohmann::json& Hyperparams::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;

  if (!done) {
    obj["epochs"] = 10;
    obj["maxBatchSize"] = 1000;

    done = true;
  }

  return obj;
}

void NeuralNetImpl::abort() {
  m_abort = true;
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj, std::istream& fin,
  const Triple& prevLayerSize) {

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
  const Triple& prevLayerSize) {

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

NeuralNetImpl::NeuralNetImpl(const Triple& inputShape, const nlohmann::json& config, Logger& logger)
  : m_logger(logger)
  , m_isTrained(false)
  , m_inputShape(inputShape)
  , m_params(getOrThrow(config, "hyperparams")) {

  Triple prevLayerSize = m_inputShape;

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

NeuralNetImpl::NeuralNetImpl(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger)
  : m_logger(logger)
  , m_isTrained(false)
  , m_inputShape(inputShape) {

  nlohmann::json paramsJson = getOrThrow(config, "hyperparams");
  nlohmann::json layersJson = getOrThrow(config, "hiddenLayers");
  nlohmann::json outLayerJson = getOrThrow(config, "outputLayer");

  m_params = Hyperparams(paramsJson);

  Triple prevLayerSize = m_inputShape;

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

void NeuralNetImpl::writeToStream(std::ostream& fout) const {
  ASSERT_MSG(m_isTrained, "Neural net is not trained");

  for (const auto& pLayer : m_layers) {
    pLayer->writeToStream(fout);
  }
}

Triple NeuralNetImpl::inputSize() const {
  return m_inputShape;
}

double NeuralNetImpl::feedForward(const Array3& x, const Vector& y) {
  const DataArray* A = &x.storage();
  for (auto& layer : m_layers) {
    layer->trainForward(*A);
    A = &layer->activations();
  }

  ConstVectorPtr outputs = Vector::createShallow(*A);

  return quadradicCost(*outputs, y);
}

OutputLayer& NeuralNetImpl::outputLayer() {
  ASSERT_MSG(!m_layers.empty(), "No output layer");
  return dynamic_cast<OutputLayer&>(*m_layers.back());
}

void NeuralNetImpl::train(LabelledDataSet& trainingData) {
  m_logger.info(STR("Epochs: " << m_params.epochs));
  m_logger.info(STR("Max batch size: " << m_params.maxBatchSize));

  const size_t N = 500; // TODO

  m_abort = false;
  for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    if (m_abort) {
      break;
    }

    m_logger.info(STR("Epoch " << epoch + 1 << "/" << m_params.epochs));
    double cost = 0.0;
    size_t samplesProcessed = 0;

    std::vector<Sample> samples;
    while (trainingData.loadSamples(samples, N) > 0) {
      [[maybe_unused]] size_t netInputSize = m_inputShape[0] * m_inputShape[1] * m_inputShape[2];
      DBG_ASSERT_MSG(samples[0].data.size() == netInputSize,
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

        m_logger.info(STR("\r  > " << samplesProcessed << "/" << m_params.maxBatchSize), false);

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
    m_logger.info(STR("\r  > cost = " << cost));

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

void NeuralNetImpl::setWeights(const std::vector<std::vector<DataArray>>& weights) {
  ASSERT(m_layers.size() == weights.size());
  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (m_layers[i]->type() == LayerType::MAX_POOLING) {
      continue;
    }

    m_layers[i]->setWeights(weights[i]);
  }
}

void NeuralNetImpl::setBiases(const std::vector<DataArray>& biases) {
  ASSERT(m_layers.size() == biases.size());
  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (m_layers[i]->type() == LayerType::MAX_POOLING) {
      continue;
    }

    m_layers[i]->setBiases(biases[i]);
  }
}

const nlohmann::json& NeuralNet::exampleConfig() {
  static nlohmann::json config;
  static bool done = false;

  if (!done) {
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

    config["hyperparams"] = Hyperparams::exampleConfig();
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

NeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger) {

  return std::make_unique<NeuralNetImpl>(inputShape, config, logger);
}

NeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger) {

  return std::make_unique<NeuralNetImpl>(inputShape, config, fin, logger);
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

