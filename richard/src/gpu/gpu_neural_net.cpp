#include "neural_net.hpp"
#include "utils.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"
#include "logger.hpp"
#include "gpu/gpu.hpp"
#include "gpu/gpu_neural_net.hpp"
#include "gpu/dense_layer.hpp"
#include "gpu/output_layer.hpp"
#include "gpu/convolutional_layer.hpp"
#include "gpu/max_pooling_layer.hpp"
#include <atomic>

namespace richard {
namespace gpu {
namespace {

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  DBG_ASSERT(actual.size() == expected.size());
  return (expected - actual).squareMagnitude() * 0.5;
};

struct StatusBuffer {
  uint32_t epoch = 0;
  uint32_t sampleIndex = 0;
};

class GpuNeuralNet : public NeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    GpuNeuralNet(const Size3& inputShape, const nlohmann::json& config, Logger& logger);
    GpuNeuralNet(const Size3& inputShape, const nlohmann::json& config, std::istream& stream,
      Logger& logger);

    CostFn costFn() const override;
    Size3 inputSize() const override;
    void writeToStream(std::ostream& stream) const override;
    void train(LabelledDataSet& data) override;
    VectorPtr evaluate(const Array3& inputs) const override;

    void abort() override;

  private:
    void initialize(const Size3& inputShape, const nlohmann::json& config, std::istream* stream);
    LayerPtr constructLayer(const nlohmann::json& obj, const Size3& prevLayerSize,
      bool isFirstLayer, std::istream* stream) const;
    void allocateGpuResources();
    void loadSampleBuffers(const LabelledDataSet& trainingData, const Sample* samples,
      size_t numSamples);
    OutputLayer& outputLayer();

    Logger& m_logger;
    bool m_isTrained;
    Size3 m_inputShape;
    size_t m_outputSize;
    Hyperparams m_params;
    GpuPtr m_gpu;
    std::vector<LayerPtr> m_layers;
    std::atomic<bool> m_abort;
    GpuBuffer m_bufferX;
    GpuBuffer m_bufferY;
    GpuBuffer m_statusBuffer;
    GpuBuffer m_costsBuffer;
    ShaderHandle m_computeCostsShader;
};

GpuNeuralNet::GpuNeuralNet(const Size3& inputShape, const nlohmann::json& config, Logger& logger)
  : m_logger(logger) {

  initialize(inputShape, config, nullptr);
}

GpuNeuralNet::GpuNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger)
  : m_logger(logger) {

  initialize(inputShape, config, &stream);
  m_isTrained = true;
}

void GpuNeuralNet::initialize(const Size3& inputShape, const nlohmann::json& config,
  std::istream* stream) {

  m_isTrained = false;
  m_inputShape = inputShape;
  m_params = Hyperparams(getOrThrow(config, "hyperparams"));
  m_gpu = createGpu(m_logger);

  Size3 prevLayerSize = m_inputShape;
  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize, m_layers.empty(), stream));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  auto outLayerJson = config["outputLayer"];
  outLayerJson["type"] = "output";
  m_layers.push_back(constructLayer(outLayerJson, prevLayerSize, false, stream));

  m_outputSize = m_layers.back()->outputSize()[0];
}

void GpuNeuralNet::abort() {
  m_abort = true;
}

LayerPtr GpuNeuralNet::constructLayer(const nlohmann::json& obj, const Size3& prevLayerSize,
  bool isFirstLayer, std::istream* stream) const {

  std::string type = getOrThrow(obj, "type");

  if (type == "dense") {
    return stream ?
      std::make_unique<DenseLayer>(*m_gpu, obj, *stream, calcProduct(prevLayerSize), isFirstLayer) :
      std::make_unique<DenseLayer>(*m_gpu, obj, calcProduct(prevLayerSize), isFirstLayer);
  }
  else if (type == "convolutional") {
    return stream ?
      std::make_unique<ConvolutionalLayer>(*m_gpu, obj, *stream, prevLayerSize, isFirstLayer) :
      std::make_unique<ConvolutionalLayer>(*m_gpu, obj, prevLayerSize, isFirstLayer);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(*m_gpu, obj, prevLayerSize);
  }
  else if (type == "output") {
    return stream ?
      std::make_unique<OutputLayer>(*m_gpu, obj, *stream, calcProduct(prevLayerSize)) :
      std::make_unique<OutputLayer>(*m_gpu, obj, calcProduct(prevLayerSize));
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

OutputLayer& GpuNeuralNet::outputLayer() {
  ASSERT_MSG(!m_layers.empty(), "No output layer");
  return dynamic_cast<OutputLayer&>(*m_layers.back());
}

NeuralNet::CostFn GpuNeuralNet::costFn() const {
  return quadradicCost;
}

void GpuNeuralNet::writeToStream(std::ostream& stream) const {
  ASSERT_MSG(m_isTrained, "Neural net is not trained");

  for (const auto& pLayer : m_layers) {
    pLayer->writeToStream(stream);
  }
}

Size3 GpuNeuralNet::inputSize() const {
  return m_inputShape;
}

void GpuNeuralNet::allocateGpuResources() {
  size_t bufferXSize = m_params.miniBatchSize * calcProduct(m_inputShape) * sizeof(netfloat_t);
  size_t bufferYSize = m_params.miniBatchSize * m_outputSize * sizeof(netfloat_t);

  GpuBufferFlags bufferFlags = GpuBufferFlags::frequentHostAccess
                             | GpuBufferFlags::large
                             | GpuBufferFlags::hostWriteAccess;

  m_bufferX = m_gpu->allocateBuffer(bufferXSize, bufferFlags);
  ASSERT_MSG(m_bufferX.data != nullptr, "Expected X buffer to be memory mapped");

  m_bufferY = m_gpu->allocateBuffer(bufferYSize, bufferFlags);
  ASSERT_MSG(m_bufferY.data != nullptr, "Expected Y buffer to be memory mapped");

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  m_statusBuffer = m_gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);
  ASSERT_MSG(m_statusBuffer.data != nullptr, "Expected status buffer to be memory mapped");

  for (LayerPtr& layer : m_layers) {
    layer->allocateGpuBuffers();
  }

  GpuBufferHandle X = m_bufferX.handle;
  for (size_t i = 0; i < m_layers.size(); ++i) {
    Layer& layer = *m_layers[i];
    const Layer* nextLayer = i + 1 == m_layers.size() ? nullptr : m_layers[i + 1].get();
    layer.createGpuShaders(X, m_statusBuffer.handle, nextLayer, m_bufferY.handle);
    X = layer.outputBuffer();
  }

  // TODO: Remove hard-coded paths
  const std::string shaderIncludesDir = "./shaders";
  const std::string computeCostsSrc = loadFile("./shaders/compute_costs.glsl");

  GpuBufferFlags costsBufferFlags = GpuBufferFlags::frequentHostAccess
                                  | GpuBufferFlags::large
                                  | GpuBufferFlags::hostReadAccess;

  m_costsBuffer = m_gpu->allocateBuffer(m_outputSize * sizeof(netfloat_t), costsBufferFlags);
  ASSERT_MSG(m_costsBuffer.data != nullptr, "Expected costs buffer to be memory mapped");

  GpuBufferBindings computeCostsBuffers{
    m_statusBuffer.handle,
    outputLayer().outputBuffer(),
    m_bufferY.handle,
    m_costsBuffer.handle
  };

  SpecializationConstants computeCostsConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_params.miniBatchSize) }
  };

  m_computeCostsShader = m_gpu->compileShader(computeCostsSrc, computeCostsBuffers,
    computeCostsConstants, { static_cast<uint32_t>(m_outputSize), 1, 1 }, { 1, 1, 1 },
    shaderIncludesDir);
}

void GpuNeuralNet::loadSampleBuffers(const LabelledDataSet& trainingData, const Sample* samples,
  size_t numSamples) {

  size_t xSize = calcProduct(m_inputShape) * sizeof(netfloat_t);
  size_t ySize = m_outputSize * sizeof(netfloat_t);

  for (size_t i = 0; i < numSamples; ++i) {
    const Sample& sample = samples[i];
    const Vector& y = trainingData.classOutputVector(sample.label);

    memcpy(m_bufferX.data + i * xSize, sample.data.data(), xSize);
    memcpy(m_bufferY.data + i * ySize, y.data(), ySize);
  }
}

void GpuNeuralNet::train(LabelledDataSet& trainingData) {
  m_logger.info(STR("Epochs: " << m_params.epochs));
  m_logger.info(STR("Batch size: " << m_params.batchSize));
  m_logger.info(STR("Mini-batch size: " << m_params.miniBatchSize));

  size_t miniBatchSize = m_params.miniBatchSize;

  ASSERT_MSG(trainingData.fetchSize() % m_params.miniBatchSize == 0,
    "Dataset fetch size must be multiple of mini-batch size");

  allocateGpuResources();

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(m_statusBuffer.data);

  m_abort = false;
  for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    if (m_abort) {
      break;
    }

    memset(m_costsBuffer.data, 0, m_costsBuffer.size);
 
    status.epoch = epoch;
    status.sampleIndex = 0;

    m_logger.info(STR("Epoch " << epoch + 1 << "/" << m_params.epochs));
    size_t samplesProcessed = 0;

    std::vector<Sample> samples;
    while (trainingData.loadSamples(samples) > 0) {
      for (size_t sampleCursor = 0; sampleCursor < samples.size(); sampleCursor += miniBatchSize) {
        loadSampleBuffers(trainingData, samples.data() + sampleCursor, miniBatchSize);

        status.sampleIndex = 0;
        for (size_t s = 0; s < miniBatchSize; ++s) {
          for (const LayerPtr& layer : m_layers) {
            layer->trainForward();
          }

          for (auto i = m_layers.crbegin(); i != m_layers.crend(); ++i) {
            (*i)->backprop();
          }

          m_gpu->queueShader(m_computeCostsShader);
        }

        for (const LayerPtr& layer : m_layers) {
          layer->updateParams();
        }

        m_gpu->flushQueue();

        samplesProcessed += miniBatchSize;
        m_logger.info(STR("\r  > " << samplesProcessed << "/" << m_params.batchSize), false);
      }

      samples.clear();

      if (samplesProcessed >= m_params.batchSize) {
        break;
      }
    }

    netfloat_t cost = 0.0;
    for (size_t i = 0; i < m_outputSize; ++i) {
      cost += reinterpret_cast<const netfloat_t*>(m_costsBuffer.data)[i];
    }
    cost /= samplesProcessed;

    m_logger.info(STR("\r  > cost = " << cost));

    trainingData.seekToBeginning();
  }

  for (LayerPtr& layer : m_layers) {
    layer->retrieveBuffers();
  }

  m_isTrained = true;
}

VectorPtr GpuNeuralNet::evaluate(const Array3&) const {
  // TODO
  return VectorPtr(nullptr);
}

}

NeuralNetPtr createNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  Logger& logger) {

  return std::make_unique<GpuNeuralNet>(inputShape, config, logger);
}

NeuralNetPtr createNeuralNet(const Size3& inputShape, const nlohmann::json& config,
  std::istream& stream, Logger& logger) {

  return std::make_unique<GpuNeuralNet>(inputShape, config, stream, logger);
}

}
}
