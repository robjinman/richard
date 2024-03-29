#include "richard/gpu/gpu.hpp"
#include "richard/gpu/gpu_neural_net.hpp"
#include "richard/gpu/dense_layer.hpp"
#include "richard/gpu/output_layer.hpp"
#include "richard/gpu/convolutional_layer.hpp"
#include "richard/gpu/max_pooling_layer.hpp"
#include "richard/neural_net.hpp"
#include "richard/event_system.hpp"
#include "richard/exception.hpp"
#include "richard/labelled_data_set.hpp"
#include "richard/logger.hpp"
#include "richard/file_system.hpp"
#include "richard/platform_paths.hpp"
#include <atomic>
#include <future>
#include <cstring>

namespace richard {
namespace gpu {
namespace {

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  DBG_ASSERT(actual.size() == expected.size());
  return (expected - actual).squareMagnitude() * netfloat_t(0.5);
};

struct StatusBuffer {
  uint32_t epoch = 0;
  uint32_t sampleIndex = 0;
};

class GpuNeuralNet : public NeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    GpuNeuralNet(const Size3& inputShape, const Config& config, EventSystem& eventSystem,
      FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger);
    GpuNeuralNet(const Size3& inputShape, const Config& config, std::istream& stream,
      EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
      Logger& logger);

    CostFn costFn() const override;
    Size3 inputSize() const override;
    void writeToStream(std::ostream& stream) const override;
    void train(LabelledDataSet& data) override;
    Vector evaluate(const Array3& inputs) const override;
    ModelDetails modelDetails() const override;

    void abort() override;

  private:
    void initialize(const Size3& inputShape, const Config& config, std::istream* stream);
    LayerPtr constructLayer(const Config& config, const Size3& prevLayerSize,
      bool isFirstLayer, std::istream* stream) const;
    void allocateGpuResources();
    void loadSampleBuffers(const LabelledDataSet& trainingData, const Sample* samples,
      size_t numSamples);
    OutputLayer& outputLayer() const;

    EventSystem& m_eventSystem;
    FileSystem& m_fileSystem;
    Logger& m_logger;
    const PlatformPaths& m_platformPaths;
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

GpuNeuralNet::GpuNeuralNet(const Size3& inputShape, const Config& config, EventSystem& eventSystem,
  FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger)
  : m_eventSystem(eventSystem)
  , m_fileSystem(fileSystem)
  , m_logger(logger)
  , m_platformPaths(platformPaths) {

  initialize(inputShape, config, nullptr);
}

GpuNeuralNet::GpuNeuralNet(const Size3& inputShape, const Config& config, std::istream& stream,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger)
  : m_eventSystem(eventSystem)
  , m_fileSystem(fileSystem)
  , m_logger(logger)
  , m_platformPaths(platformPaths) {

  initialize(inputShape, config, &stream);
  m_isTrained = true;
}

void GpuNeuralNet::initialize(const Size3& inputShape, const Config& config,
  std::istream* stream) {

  m_isTrained = false;
  m_inputShape = inputShape;
  m_params = Hyperparams(config.getObject("hyperparams"));
  m_gpu = createGpu(m_logger, config.contains("gpu") ? config.getObject("gpu") : Config{});

  Size3 prevLayerSize = m_inputShape;
  if (config.contains("hiddenLayers")) {
    auto layersConfig = config.getObjectArray("hiddenLayers");

    for (auto layerConfig : layersConfig) {
      m_layers.push_back(constructLayer(layerConfig, prevLayerSize, m_layers.empty(), stream));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  auto outLayerConfig = config.getObject("outputLayer");
  outLayerConfig.setString("type", "output");
  m_layers.push_back(constructLayer(outLayerConfig, prevLayerSize, false, stream));

  m_outputSize = m_layers.back()->outputSize()[0];

  allocateGpuResources();
}

ModelDetails GpuNeuralNet::modelDetails() const {
  return ModelDetails{
    { "Batch size", std::to_string(m_params.batchSize) },
    { "Mini-batch size", std::to_string(m_params.miniBatchSize) },
    { "Epochs", std::to_string(m_params.epochs) }
  };
}

void GpuNeuralNet::abort() {
  m_abort = true;
}

LayerPtr GpuNeuralNet::constructLayer(const Config& config, const Size3& prevLayerSize,
  bool isFirstLayer, std::istream* stream) const {

  auto type = config.getString("type");

  if (type == "dense") {
    return stream ?
      std::make_unique<DenseLayer>(*m_gpu, m_fileSystem, m_platformPaths, config, *stream,
        calcProduct(prevLayerSize), isFirstLayer) :
      std::make_unique<DenseLayer>(*m_gpu, m_fileSystem, m_platformPaths, config,
        calcProduct(prevLayerSize), isFirstLayer);
  }
  else if (type == "convolutional") {
    return stream ?
      std::make_unique<ConvolutionalLayer>(*m_gpu, m_fileSystem, m_platformPaths, config, *stream,
        prevLayerSize, isFirstLayer) :
      std::make_unique<ConvolutionalLayer>(*m_gpu, m_fileSystem, m_platformPaths, config,
        prevLayerSize, isFirstLayer);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(*m_gpu, m_fileSystem, m_platformPaths, config,
      prevLayerSize);
  }
  else if (type == "output") {
    return stream ?
      std::make_unique<OutputLayer>(*m_gpu, m_fileSystem, m_platformPaths, config, *stream,
        calcProduct(prevLayerSize)) :
      std::make_unique<OutputLayer>(*m_gpu, m_fileSystem, m_platformPaths, config,
        calcProduct(prevLayerSize));
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

OutputLayer& GpuNeuralNet::outputLayer() const {
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

  GpuBufferFlags costsBufferFlags = GpuBufferFlags::frequentHostAccess
                                  | GpuBufferFlags::large
                                  | GpuBufferFlags::hostReadAccess;

  m_costsBuffer = m_gpu->allocateBuffer(m_outputSize * sizeof(netfloat_t), costsBufferFlags);
  ASSERT_MSG(m_costsBuffer.data != nullptr, "Expected costs buffer to be memory mapped");

  GpuBufferBindings computeCostsBuffers{
    { m_statusBuffer.handle, BufferAccessMode::write },
    { outputLayer().outputBuffer(), BufferAccessMode::read },
    { m_bufferY.handle, BufferAccessMode::read },
    { m_costsBuffer.handle, BufferAccessMode::write }
  };

  SpecializationConstants computeCostsConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_params.miniBatchSize) }
  };

  std::string computeCostsShaderName = "compute_costs.spv";
  auto computeCostsShaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders",
    computeCostsShaderName));

  m_computeCostsShader = m_gpu->addShader(computeCostsShaderName, computeCostsShaderCode,
    computeCostsBuffers, computeCostsConstants, 0, { static_cast<uint32_t>(m_outputSize), 1, 1 });
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
  uint32_t miniBatchSize = m_params.miniBatchSize;

  ASSERT_MSG(trainingData.fetchSize() % m_params.miniBatchSize == 0,
    "Dataset fetch size must be multiple of mini-batch size");

  ASSERT_MSG(m_params.batchSize % m_params.miniBatchSize == 0,
    "Batch size must be multiple of mini-batch size");

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(m_statusBuffer.data);

  m_abort = false;
  for (uint32_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    if (m_abort) {
      break;
    }

    m_eventSystem.raise(EEpochStarted{epoch, m_params.epochs});

    memset(m_costsBuffer.data, 0, m_costsBuffer.size);
 
    status.epoch = epoch;
    status.sampleIndex = 0;

    uint32_t samplesProcessed = 0;

    auto pendingSamples = std::async([&]() { return trainingData.loadSamples(); });
    std::vector<Sample> samples = pendingSamples.get();

    while (samples.size() > 0) {
      pendingSamples = std::async([&]() { return trainingData.loadSamples(); });

      for (size_t sampleCursor = 0; sampleCursor < samples.size(); sampleCursor += miniBatchSize) {
        loadSampleBuffers(trainingData, samples.data() + sampleCursor, miniBatchSize);

        status.sampleIndex = 0;
        for (uint32_t s = 0; s < miniBatchSize; ++s) {
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
        m_eventSystem.raise(ESampleProcessed{samplesProcessed - 1, m_params.batchSize});

        if (samplesProcessed >= m_params.batchSize) {
          break;
        }
      }

      if (samplesProcessed >= m_params.batchSize) {
        break;
      }

      samples = pendingSamples.get();
    }

    netfloat_t cost = 0.0;
    for (size_t i = 0; i < m_outputSize; ++i) {
      cost += reinterpret_cast<const netfloat_t*>(m_costsBuffer.data)[i];
    }
    cost /= samplesProcessed;

    m_eventSystem.raise(EEpochCompleted{epoch, m_params.epochs, cost});

    trainingData.seekToBeginning();
  }

  for (LayerPtr& layer : m_layers) {
    layer->retrieveBuffers();
  }

  m_isTrained = true;
}

Vector GpuNeuralNet::evaluate(const Array3& sample) const {
  memcpy(m_bufferX.data, sample.data(), sample.size() * sizeof(netfloat_t));

  for (const LayerPtr& layer : m_layers) {
    layer->evalForward();
  }

  m_gpu->flushQueue();

  return outputLayer().activations();
}

}

NeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger) {

  return std::make_unique<GpuNeuralNet>(inputShape, config, eventSystem, fileSystem, platformPaths,
    logger);
}

NeuralNetPtr createNeuralNet(const Size3& inputShape, const Config& config, std::istream& stream,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger) {

  return std::make_unique<GpuNeuralNet>(inputShape, config, stream, eventSystem, fileSystem,
    platformPaths, logger);
}

}
}
