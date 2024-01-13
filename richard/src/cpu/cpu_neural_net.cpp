#include "cpu/dense_layer.hpp"
#include "cpu/max_pooling_layer.hpp"
#include "cpu/convolutional_layer.hpp"
#include "cpu/output_layer.hpp"
#include "cpu/cpu_neural_net.hpp"
#include "utils.hpp"
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

const NeuralNet::CostFn quadraticCost = [](const Vector& actual, const Vector& expected) {
  DBG_ASSERT(actual.size() == expected.size());
  return (expected - actual).squareMagnitude() * 0.5;
};

class CpuNeuralNetImpl : public CpuNeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    CpuNeuralNetImpl(const Size3& inputShape, const nlohmann::json& config, Logger& logger);
    CpuNeuralNetImpl(const Size3& inputShape, const nlohmann::json& config, std::istream& stream,
      Logger& logger);

    CostFn costFn() const override;
    Size3 inputSize() const override;
    void writeToStream(std::ostream& s) const override;
    void train(LabelledDataSet& data) override;
    VectorPtr evaluate(const Array3& inputs) const override;

    void abort() override;

    // Exposed for testing
    //
    Layer& test_getLayer(size_t index) override;

  private:
    void initialize(const Size3& inputShape, const nlohmann::json& config, std::istream* stream);
    LayerPtr constructLayer(const nlohmann::json& obj, const Size3& prevLayerSize,
      std::istream* stream) const;
    netfloat_t feedForward(const Array3& x, const Vector& y);
    void backPropagate(const Array3& x, const Vector& y);
    void updateParams(size_t epoch);

    Logger& m_logger;
    bool m_isTrained;
    Size3 m_inputShape;
    Hyperparams m_params;
    std::vector<LayerPtr> m_layers;
    std::atomic<bool> m_abort;
};

CpuNeuralNetImpl::CpuNeuralNetImpl(const Size3& inputShape, const nlohmann::json& config,
  Logger& logger)
  : m_logger(logger) {

  initialize(inputShape, config, nullptr);
}

CpuNeuralNetImpl::CpuNeuralNetImpl(const Size3& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger)
  : m_logger(logger) {

  initialize(inputShape, config, &stream);
  m_isTrained = true;
}

void CpuNeuralNetImpl::initialize(const Size3& inputShape, const nlohmann::json& config,
  std::istream* stream) {

  m_isTrained = false;
  m_inputShape = inputShape;
  m_params = Hyperparams(getOrThrow(config, "hyperparams"));

  Size3 prevLayerSize = m_inputShape;

  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize, stream));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  auto outLayerJson = getOrThrow(config, "outputLayer");
  outLayerJson["type"] = "output";
  m_layers.push_back(constructLayer(outLayerJson, prevLayerSize, stream));
}

LayerPtr CpuNeuralNetImpl::constructLayer(const nlohmann::json& obj, const Size3& prevLayerSize,
  std::istream* stream) const {

  std::string type = getOrThrow(obj, "type");

  if (type == "dense") {
    return stream ?
      std::make_unique<DenseLayer>(obj, *stream, calcProduct(prevLayerSize)) :
      std::make_unique<DenseLayer>(obj, calcProduct(prevLayerSize));
  }
  else if (type == "convolutional") {
    return stream ?
      std::make_unique<ConvolutionalLayer>(obj, *stream, prevLayerSize) :
      std::make_unique<ConvolutionalLayer>(obj, prevLayerSize);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize);
  }
  else if (type == "output") {
    return stream ?
      std::make_unique<OutputLayer>(obj, *stream, calcProduct(prevLayerSize)) :
      std::make_unique<OutputLayer>(obj, calcProduct(prevLayerSize));
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

void CpuNeuralNetImpl::abort() {
  m_abort = true;
}

NeuralNet::CostFn CpuNeuralNetImpl::costFn() const {
  return quadraticCost;
}

void CpuNeuralNetImpl::writeToStream(std::ostream& stream) const {
  ASSERT_MSG(m_isTrained, "Neural net is not trained");

  for (const auto& pLayer : m_layers) {
    pLayer->writeToStream(stream);
  }
}

Size3 CpuNeuralNetImpl::inputSize() const {
  return m_inputShape;
}

netfloat_t CpuNeuralNetImpl::feedForward(const Array3& x, const Vector& y) {
  const DataArray* A = &x.storage();
  for (auto& layer : m_layers) {
    layer->trainForward(*A);
    A = &layer->activations();
  }

  ConstVectorPtr outputs = Vector::createShallow(*A);

  return quadraticCost(*outputs, y);
}

void CpuNeuralNetImpl::backPropagate(const Array3& x, const Vector& y) {
  int numLayers = static_cast<int>(m_layers.size());

  for (int i = numLayers - 1; i >= 0; --i) {
    if (i == numLayers - 1) {
      m_layers[i]->updateDeltas(m_layers[i - 1]->activations(), y.storage());
    }
    else if (i == 0) {
      m_layers[i]->updateDeltas(x.storage(), m_layers[i + 1]->inputDelta());
    }
    else {
      m_layers[i]->updateDeltas(m_layers[i - 1]->activations(), m_layers[i + 1]->inputDelta());
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
      DBG_ASSERT_MSG(samples[0].data.size() == calcProduct(m_inputShape),
        "Sample size is " << samples[0].data.size() << ", expected " << calcProduct(m_inputShape));

      for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const Array3& x = sample.data;
        const Vector& y = trainingData.classOutputVector(sample.label);

        cost += feedForward(x, y);
        backPropagate(x, y);

        ++samplesProcessed;

        bool lastSample = samplesProcessed == m_params.batchSize;
        if ((samplesProcessed % m_params.miniBatchSize == 0) || lastSample) {
          updateParams(epoch);
        }

        m_logger.info(STR("\r  > " << samplesProcessed << "/" << m_params.batchSize), false);

        if (samplesProcessed >= m_params.batchSize) {
          break;
        }
      }

      samples.clear();

      if (samplesProcessed >= m_params.batchSize) {
        break;
      }
    }

    cost /= samplesProcessed;
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

CpuNeuralNetPtr createNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  Logger& logger) {

  return std::make_unique<CpuNeuralNetImpl>(inputShape, config, logger);
}

CpuNeuralNetPtr createNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger) {

  return std::make_unique<CpuNeuralNetImpl>(inputShape, config, stream, logger);
}

}
}
