#include "mock_logger.hpp"
#include "mock_gpu_layer.hpp"
#include "mock_cpu_layer.hpp"
#include <cpu/max_pooling_layer.hpp>
#include <gpu/max_pooling_layer.hpp>
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

class GpuMaxPoolingLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

void cpuMaxPoolingLayerTrainForward(const Config& config, const Array3& inputs,
  Array3& activations, Array3& mask) {

  cpu::MaxPoolingLayer layer(config, inputs.shape());

  layer.trainForward(inputs.storage());

  mask = layer.test_mask();
  activations = Array3(layer.activations(), { 2, 2, 2 });
}

TEST_F(GpuMaxPoolingLayerTest, trainForward) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  Array3 inputs({
    {
      { 6, 0, 1, 2 },
      { 5, 5, 6, 7 },
      { 3, 8, 7, 6 },
      { 2, 6, 3, 1 }
    }, {
      { 9, 5, 4, 3 },
      { 7, 2, 1, 0 },
      { 4, 1, 2, 5 },
      { 2, 8, 4, 6 }
    }
  });

  size_t inputBufferSize = inputs.size() * sizeof(netfloat_t);

  GpuBufferFlags inputBufferFlags = GpuBufferFlags::large
                                  | GpuBufferFlags::hostWriteAccess;

  GpuBuffer inputBuffer = gpu->allocateBuffer(inputBufferSize, inputBufferFlags);

  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  Config config;
  config.setIntegerArray<size_t>("regionSize", { 2, 2 });

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::MaxPoolingLayer layer(*gpu, *fileSystem, *platformPaths, config, { 4, 4, 2 });

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, deltaBuffer).WillByDefault(testing::Return(0));

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, &nextLayer, 0);

  layer.trainForward();
  gpu->flushQueue();

  Array3 A(2, 2, 2);
  gpu->retrieveBuffer(layer.outputBuffer(), A.data());

  Array3 mask(4, 4, 2);
  gpu->retrieveBuffer(layer.test_maskBuffer(), mask.data());

  Array3 expectedA;
  Array3 expectedMask;
  cpuMaxPoolingLayerTrainForward(config, inputs, expectedA, expectedMask);

  EXPECT_EQ(A.shape(), expectedA.shape());

  for (size_t k = 0; k < A.D(); ++k) {
    for (size_t j = 0; j < A.H(); ++j) {
      for (size_t i = 0; i < A.W(); ++i) {
        EXPECT_NEAR(A.at(i, j, k), expectedA.at(i, j, k), FLOAT_TOLERANCE);
      }
    }
  }

  EXPECT_EQ(mask.shape(), expectedMask.shape());

  for (size_t k = 0; k < mask.D(); ++k) {
    for (size_t j = 0; j < mask.H(); ++j) {
      for (size_t i = 0; i < mask.W(); ++i) {
        EXPECT_NEAR(mask.at(i, j, k), expectedMask.at(i, j, k), FLOAT_TOLERANCE);
      }
    }
  }
}

void cpuMaxPoolingLayerBackprop(const Config& config, const Array3& mask,
  const Array3& outputDelta, Array3& inputDelta) {

  cpu::MaxPoolingLayer layer(config, { 4, 4, 2 });

  layer.test_setMask(mask);
  layer.updateDeltas(DataArray(), outputDelta.storage());
  
  inputDelta = Array3(layer.inputDelta(), 4, 4, 2);
}

TEST_F(GpuMaxPoolingLayerTest, backprop) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  Array3 mask({
    {
      { 0, 0, 0, 1 },
      { 1, 0, 0, 0 },
      { 0, 1, 0, 0 },
      { 0, 0, 0, 1 }
    },{
      { 0, 1, 0, 0 },
      { 0, 0, 1, 0 },
      { 0, 0, 0, 1 },
      { 1, 0, 0, 0 }
    }
  });

  Array3 deltaA({
    {
      { 3, 6 },
      { 4, 2 }
    }, {
      { 5, 1 },
      { 9, 8 }
    }
  });

  GpuBuffer deltaABuffer = gpu->allocateBuffer(deltaA.size() * sizeof(netfloat_t),
    GpuBufferFlags::large);

  gpu->submitBufferData(deltaABuffer.handle, deltaA.data());

  Config config;
  config.setIntegerArray<size_t>("regionSize", { 2, 2 });

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::MaxPoolingLayer layer(*gpu, *fileSystem, *platformPaths, config, { 4, 4, 2 });

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, inputDeltaBuffer).WillByDefault(testing::Return(deltaABuffer.handle));

  layer.allocateGpuBuffers();

  gpu->submitBufferData(layer.test_maskBuffer(), mask.data());

  layer.createGpuShaders(0, 0, &nextLayer, 0);

  layer.backprop();
  gpu->flushQueue();

  Array3 inputDelta(4, 4, 2);
  gpu->retrieveBuffer(layer.inputDeltaBuffer(), inputDelta.data());

  Array3 expectedInputDelta;
  cpuMaxPoolingLayerBackprop(config, mask, deltaA, expectedInputDelta);

  EXPECT_EQ(inputDelta.shape(), expectedInputDelta.shape());

  for (size_t k = 0; k < inputDelta.D(); ++k) {
    for (size_t j = 0; j < inputDelta.H(); ++j) {
      for (size_t i = 0; i < inputDelta.W(); ++i) {
        EXPECT_NEAR(inputDelta.at(i, j, k), expectedInputDelta.at(i, j, k), FLOAT_TOLERANCE);
      }
    }
  }
}
