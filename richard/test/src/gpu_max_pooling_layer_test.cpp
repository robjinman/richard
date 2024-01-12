#include "mock_logger.hpp"
#include "mock_gpu_layer.hpp"
#include "mock_cpu_layer.hpp"
#include <cpu/max_pooling_layer.hpp>
#include <gpu/max_pooling_layer.hpp>
#include <gpu/gpu.hpp>
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

void cpuMaxPoolingLayerTrainForward(const nlohmann::json& config, const Array3& inputs,
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

  nlohmann::json config;
  config["regionSize"] = { 2, 2 };

  gpu::MaxPoolingLayer layer(*gpu, config, { 4, 4, 2 });

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
/*
void cpuMaxPoolingLayerBackprop(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs, const Vector& dA, Matrix& deltaW, Vector& deltaB) {

  cpu::MaxPoolingLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());
  layer.updateDeltas(inputs.storage(), dA.storage());

  deltaW = layer.test_deltaW();
  deltaB = layer.test_deltaB();
}

TEST_F(GpuMaxPoolingLayerTest, backprop) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  const size_t layerInputSize = 4;
  const size_t layerSize = 2;

  size_t inputBufferSize = layerInputSize * sizeof(netfloat_t);

  GpuBufferFlags inputBufferFlags = GpuBufferFlags::large
                                  | GpuBufferFlags::hostWriteAccess;

  GpuBuffer inputBuffer = gpu->allocateBuffer(inputBufferSize, inputBufferFlags);

  Vector inputs{ 0.5, 0.4, 0.3, 0.2 };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  Vector nextDelta({ 0.2, 0.7 });
  Matrix nextW({
    { 0.2, 0.5 },
    { 0.4, 0.3 }
  });

  Vector dA = nextW.transposeMultiply(nextDelta);

  GpuBufferFlags bufferFlags = GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess;

  GpuBuffer nextBufferW = gpu->allocateBuffer(nextW.size() * sizeof(netfloat_t), bufferFlags);
  GpuBuffer nextBufferD = gpu->allocateBuffer(nextDelta.size() * sizeof(netfloat_t), bufferFlags);

  gpu->submitBufferData(nextBufferW.handle, nextW.data());
  gpu->submitBufferData(nextBufferD.handle, nextDelta.data());

  nlohmann::json config;
  config["size"] = layerSize;
  config["learnRate"] = 0.1;
  config["learnRateDecay"] = 1.0;
  config["dropoutRate"] = 0.0;

  gpu::MaxPoolingLayer layer(*gpu, config, layerInputSize, true);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, weightsBuffer).WillByDefault(testing::Return(nextBufferW.handle));
  ON_CALL(nextLayer, deltaBuffer).WillByDefault(testing::Return(nextBufferD.handle));
  ON_CALL(nextLayer, size).WillByDefault(testing::Return(nextDelta.size()));

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, &nextLayer, 0);

  layer.trainForward();
  layer.backprop();

  gpu->flushQueue();

  Vector delta(layerSize);
  Matrix deltaW(W.cols(), W.rows());
  Vector deltaB(B.size());

  gpu->retrieveBuffer(layer.deltaBuffer(), delta.data());
  gpu->retrieveBuffer(layer.test_deltaWBuffer(), deltaW.data());
  gpu->retrieveBuffer(layer.test_deltaBBuffer(), deltaB.data());

  Matrix expectedDeltaW;
  Vector expectedDeltaB;

  cpuMaxPoolingLayerBackprop(config, W, B, inputs, dA, expectedDeltaW, expectedDeltaB);

  for (size_t j = 0; j < deltaW.rows(); ++j) {
    for (size_t i = 0; i < deltaW.cols(); ++i) {
      EXPECT_NEAR(deltaW.at(i, j), expectedDeltaW.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < deltaB.size(); ++i) {
    EXPECT_NEAR(deltaB[i], expectedDeltaB[i], FLOAT_TOLERANCE);
  }
}

TEST_F(GpuMaxPoolingLayerTest, updateParams) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  const size_t layerInputSize = 4;
  const size_t layerSize = 2;

  size_t inputBufferSize = layerInputSize * sizeof(netfloat_t);

  GpuBufferFlags inputBufferFlags = GpuBufferFlags::large
                                  | GpuBufferFlags::hostWriteAccess;

  GpuBuffer inputBuffer = gpu->allocateBuffer(inputBufferSize, inputBufferFlags);

  Vector inputs{ 0.5, 0.4, 0.3, 0.2 };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  nlohmann::json config;
  config["size"] = layerSize;
  config["learnRate"] = 0.1;
  config["learnRateDecay"] = 1.0;
  config["dropoutRate"] = 0.0;

  gpu::MaxPoolingLayer layer(*gpu, config, layerInputSize, true);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  testing::NiceMock<MockGpuLayer> nextLayer;

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, &nextLayer, 0);

  Matrix deltaW({
    { 0.5, 0.3, 0.7, 0.1 },
    { 0.8, 0.6, 0.2, 0.9 }
  });

  Vector deltaB({ 0.5, 0.1 });

  gpu->submitBufferData(layer.test_deltaWBuffer(), deltaW.data());
  gpu->submitBufferData(layer.test_deltaBBuffer(), deltaB.data());

  layer.updateParams();

  gpu->flushQueue();

  layer.retrieveBuffers();

  Matrix expectedW = W - deltaW * 0.1;
  Vector expectedB = B - deltaB * 0.1;

  const Matrix& actualW = layer.test_W();
  const Vector& actualB = layer.test_B();

  EXPECT_EQ(expectedW.cols(), actualW.cols());
  EXPECT_EQ(expectedW.rows(), actualW.rows());

  for (size_t j = 0; j < expectedW.rows(); ++j) {
    for (size_t i = 0; i < expectedW.cols(); ++i) {
      EXPECT_NEAR(actualW.at(i, j), expectedW.at(i, j), FLOAT_TOLERANCE);
    }
  }

  EXPECT_EQ(expectedB.size(), actualB.size());

  for (size_t i = 0; i < expectedB.size(); ++i) {
    EXPECT_NEAR(actualB[i], expectedB[i], FLOAT_TOLERANCE);
  }
}
*/