#include "cpu/dense_layer.hpp"
#include "cpu/max_pooling_layer.hpp"
#include "cpu/convolutional_layer.hpp"
#include "cpu/output_layer.hpp"
#include "cpu/cpu_neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"
#include "logger.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <atomic>

namespace richard {
namespace cpu {
namespace {

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  DBG_ASSERT(actual.size() == expected.size());
  return (expected - actual).squareMagnitude() * 0.5;
};

class CpuNeuralNetImpl : public CpuNeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    CpuNeuralNetImpl(const Triple& inputShape, const nlohmann::json& config, Logger& logger);
    CpuNeuralNetImpl(const Triple& inputShape, const nlohmann::json& config, std::istream& s,
      Logger& logger);

    CostFn costFn() const override;
    Triple inputSize() const override;
    void writeToStream(std::ostream& s) const override;
    void train(LabelledDataSet& data) override;
    VectorPtr evaluate(const Array3& inputs) const override;

    void abort() override;

    // Exposed for testing
    //
    Layer& test_getLayer(size_t index) override;

  private:
    netfloat_t feedForward(const Array3& x, const Vector& y);
    void backPropagate(const Array3& x, const Vector& y);
    void updateParams(size_t epoch);
    OutputLayer& outputLayer();

    Logger& m_logger;
    bool m_isTrained;
    Triple m_inputShape;
    Hyperparams m_params;
    std::vector<LayerPtr> m_layers;
    std::atomic<bool> m_abort;
};

void CpuNeuralNetImpl::abort() {
  m_abort = true;
}

LayerPtr constructLayer(const nlohmann::json& obj, std::istream& stream,
  const Triple& prevLayerSize) {
  
  size_t numInputs = prevLayerSize[0] * prevLayerSize[1] * prevLayerSize[2];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, stream, numInputs);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, stream, prevLayerSize[0], prevLayerSize[1],
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

LayerPtr constructLayer(const nlohmann::json& obj, const Triple& prevLayerSize) {
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

CpuNeuralNetImpl::CpuNeuralNetImpl(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger)
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

CpuNeuralNetImpl::CpuNeuralNetImpl(const Triple& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger)
  : m_logger(logger)
  , m_isTrained(false)
  , m_inputShape(inputShape)
  , m_params(getOrThrow(config, "hyperparams")) {

  nlohmann::json layersJson = getOrThrow(config, "hiddenLayers");
  nlohmann::json outLayerJson = getOrThrow(config, "outputLayer");

  Triple prevLayerSize = m_inputShape;

  for (auto& layerJson : layersJson) {
    m_layers.push_back(constructLayer(layerJson, stream, prevLayerSize));
    prevLayerSize = m_layers.back()->outputSize();
  }
  m_layers.push_back(std::make_unique<OutputLayer>(outLayerJson, stream,
    prevLayerSize[0] * prevLayerSize[1] * prevLayerSize[2]));

  m_isTrained = true;
}

NeuralNet::CostFn CpuNeuralNetImpl::costFn() const {
  return quadradicCost;
}

void CpuNeuralNetImpl::writeToStream(std::ostream& stream) const {
  ASSERT_MSG(m_isTrained, "Neural net is not trained");

  for (const auto& pLayer : m_layers) {
    pLayer->writeToStream(stream);
  }
}

Triple CpuNeuralNetImpl::inputSize() const {
  return m_inputShape;
}

netfloat_t CpuNeuralNetImpl::feedForward(const Array3& x, const Vector& y) {
  const DataArray* A = &x.storage();
  for (auto& layer : m_layers) {
    layer->trainForward(*A);
    A = &layer->activations();
  }

  ConstVectorPtr outputs = Vector::createShallow(*A);

  return quadradicCost(*outputs, y);
}

OutputLayer& CpuNeuralNetImpl::outputLayer() {
  ASSERT_MSG(!m_layers.empty(), "No output layer");
  return dynamic_cast<OutputLayer&>(*m_layers.back());
}

void CpuNeuralNetImpl::backPropagate(const Array3& x, const Vector& y) {
  for (int l = static_cast<int>(m_layers.size()) - 1; l >= 0; --l) {
    if (l == static_cast<int>(m_layers.size()) - 1) {
      outputLayer().updateDelta(m_layers[m_layers.size() - 2]->activations(), y.storage());
    }
    else if (l == 0) {
      m_layers[l]->updateDelta(x.storage(), *m_layers[l + 1]);
    }
    else {
      m_layers[l]->updateDelta(m_layers[l - 1]->activations(), *m_layers[l + 1]);
    }
  }
}

void CpuNeuralNetImpl::updateParams(size_t epoch) { 
  for (auto& layer : m_layers) {
    layer->updateParams(epoch);
  }
}

void CpuNeuralNetImpl::train(LabelledDataSet& trainingData) {
  m_logger.info(STR("Epochs: " << m_params.epochs));
  m_logger.info(STR("Batch size: " << m_params.batchSize));
  m_logger.info(STR("Mini-batch size: " << m_params.miniBatchSize));

  m_abort = false;
  for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    if (m_abort) {
      break;
    }

    m_logger.info(STR("Epoch " << epoch + 1 << "/" << m_params.epochs));
    netfloat_t cost = 0.0;
    size_t samplesProcessed = 0;

    std::vector<Sample> samples;
    while (trainingData.loadSamples(samples) > 0) {
      [[maybe_unused]] size_t netInputSize = m_inputShape[0] * m_inputShape[1] * m_inputShape[2];
      DBG_ASSERT_MSG(samples[0].data.size() == netInputSize,
        "Sample size is " << samples[0].data.size() << ", expected " << netInputSize);

      for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const Array3& x = sample.data;
        const Vector& y = trainingData.classOutputVector(sample.label);

        cost += feedForward(x, y);
        backPropagate(x, y);

        bool lastSample = samplesProcessed + 1 == m_params.batchSize;
        if (((samplesProcessed + 1) % m_params.miniBatchSize == 0) || lastSample) {
          updateParams(epoch);
        }

        m_logger.info(STR("\r  > " << samplesProcessed << "/" << m_params.batchSize), false);

        ++samplesProcessed;
        if (samplesProcessed >= m_params.batchSize) {
          break;
        }
      }

      samples.clear();

      if (samplesProcessed >= m_params.batchSize) {
        break;
      }
    }

    cost = cost / samplesProcessed;
    m_logger.info(STR("\r  > cost = " << cost));

    trainingData.seekToBeginning();
  }

  m_isTrained = true;
}

VectorPtr CpuNeuralNetImpl::evaluate(const Array3& x) const {
  DataArray A;

  for (size_t i = 0; i < m_layers.size(); ++i) {
    A = m_layers[i]->evalForward(i == 0 ? x.storage() : A);
  }

  return std::make_unique<Vector>(A);
}

}

Layer& CpuNeuralNetImpl::test_getLayer(size_t index) {
  ASSERT(index < m_layers.size());
  return *m_layers[index];
}

CpuNeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger) {

  return std::make_unique<CpuNeuralNetImpl>(inputShape, config, logger);
}

CpuNeuralNetPtr createNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger) {

  return std::make_unique<CpuNeuralNetImpl>(inputShape, config, stream, logger);
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

}
}
