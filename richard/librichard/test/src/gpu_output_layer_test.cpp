#include "mock_logger.hpp"
#include <cpu/output_layer.hpp>
#include <gpu/output_layer.hpp>
#include <gpu/gpu.hpp>
#include <file_system.hpp>
#include <platform_paths.hpp>
#include <gtest/gtest.h>

using namespace richard;

using richard::gpu::GpuPtr;
using richard::gpu::GpuBuffer;
using richard::gpu::GpuBufferFlags;

const double FLOAT_TOLERANCE = 0.0001;

struct StatusBuffer {
  uint32_t epoch;
  uint32_t sampleIndex;
};

class GpuOutputLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

Vector cpuOutputLayerTrainForward(const Config& config, const Matrix& W, const Vector& B,
  const Vector& inputs) {

  cpu::OutputLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());

  return layer.activations();
}

TEST_F(GpuOutputLayerTest, trainForward) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t miniBatchSize = 1;
  const size_t layerInputSize = 4;
  const size_t outputSize = 2;

  size_t bufferYSize = miniBatchSize * outputSize * sizeof(netfloat_t);

  GpuBufferFlags bufferYFlags = GpuBufferFlags::frequentHostAccess
                              | GpuBufferFlags::large
                              | GpuBufferFlags::hostWriteAccess;

  GpuBuffer bufferY = gpu->allocateBuffer(bufferYSize, bufferYFlags);

  size_t inputBufferSize = layerInputSize * sizeof(netfloat_t);

  GpuBufferFlags inputBufferFlags = GpuBufferFlags::large
                                  | GpuBufferFlags::hostWriteAccess;

  GpuBuffer inputBuffer = gpu->allocateBuffer(inputBufferSize, inputBufferFlags);

  Vector inputs{ 0.5f, 0.4f, 0.3f, 0.2f };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  Config config;
  config.setNumber("size", outputSize);
  config.setNumber("learnRate", 0.1);
  config.setNumber("learnRateDecay", 1.0);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::OutputLayer layer(*gpu, *fileSystem, *platformPaths, config, layerInputSize);

  Matrix W({
    { 0.1f, 0.2f, 0.3f, 0.4f },
    { 0.5f, 0.4f, 0.3f, 0.2f }
  });

  Vector B({ 0.7f, 0.8f });

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, nullptr, bufferY.handle);

  layer.trainForward();
  gpu->flushQueue();

  Vector A(outputSize);
  gpu->retrieveBuffer(layer.outputBuffer(), A.data());

  Vector expectedA = cpuOutputLayerTrainForward(config, W, B, inputs);

  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_NEAR(A[i], expectedA[i], FLOAT_TOLERANCE);
  }
}

void cpuOutputLayerBackprop(const Config& config, const Matrix& W, const Vector& B,
  const Vector& inputs, const Vector& Y, Matrix& deltaW, Vector& deltaB) {

  cpu::OutputLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());
  layer.updateDeltas(inputs.storage(), Y.storage());

  deltaW = layer.test_deltaW();
  deltaB = layer.test_deltaB();
}

TEST_F(GpuOutputLayerTest, backprop) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  const size_t layerInputSize = 4;
  const size_t outputSize = 2;

  Vector Y({ 0.f, 1.f });

  GpuBufferFlags bufferYFlags = GpuBufferFlags::frequentHostAccess
                              | GpuBufferFlags::large
                              | GpuBufferFlags::hostWriteAccess;

  GpuBuffer bufferY = gpu->allocateBuffer(Y.size() * sizeof(netfloat_t), bufferYFlags);
  ASSERT_NE(bufferY.data, nullptr);

  memcpy(bufferY.data, Y.data(), Y.size() * sizeof(netfloat_t));

  size_t inputBufferSize = layerInputSize * sizeof(netfloat_t);

  GpuBufferFlags inputBufferFlags = GpuBufferFlags::large
                                  | GpuBufferFlags::hostWriteAccess;

  GpuBuffer inputBuffer = gpu->allocateBuffer(inputBufferSize, inputBufferFlags);

  Vector inputs{ 0.5f, 0.4f, 0.3f, 0.2f };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  Config config;
  config.setNumber("size", outputSize);
  config.setNumber("learnRate", 0.1);
  config.setNumber("learnRateDecay", 1.0);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::OutputLayer layer(*gpu, *fileSystem, *platformPaths, config, layerInputSize);

  Matrix W({
    { 0.1f, 0.2f, 0.3f, 0.4f },
    { 0.5f, 0.4f, 0.3f, 0.2f }
  });

  Vector B({ 0.7f, 0.8f });

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, nullptr, bufferY.handle);

  layer.trainForward();
  layer.backprop();

  gpu->flushQueue();

  Matrix deltaW(W.cols(), W.rows());
  Vector deltaB(B.size());

  gpu->retrieveBuffer(layer.test_deltaWBuffer(), deltaW.data());
  gpu->retrieveBuffer(layer.test_deltaBBuffer(), deltaB.data());

  Matrix expectedDeltaW;
  Vector expectedDeltaB;

  cpuOutputLayerBackprop(config, W, B, inputs, Y, expectedDeltaW, expectedDeltaB);

  for (size_t j = 0; j < deltaW.rows(); ++j) {
    for (size_t i = 0; i < deltaW.cols(); ++i) {
      EXPECT_NEAR(deltaW.at(i, j), expectedDeltaW.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < deltaB.size(); ++i) {
    EXPECT_NEAR(deltaB[i], expectedDeltaB[i], FLOAT_TOLERANCE);
  }
}
