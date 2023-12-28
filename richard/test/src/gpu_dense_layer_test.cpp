#include "mock_logger.hpp"
#include "mock_gpu_layer.hpp"
#include "mock_cpu_layer.hpp"
#include <cpu/dense_layer.hpp>
#include <gpu/dense_layer.hpp>
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

class GpuDenseLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

Vector cpuDenseLayerTrainForward(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs) {

  cpu::DenseLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());

  return layer.activations();
}

TEST_F(GpuDenseLayerTest, trainForward) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t miniBatchSize = 4;
  const size_t layerSize = 2;
  const size_t layerInputSize = 4;
  const size_t networkOutputSize = 2;

  size_t bufferYSize = miniBatchSize * networkOutputSize * sizeof(netfloat_t);

  GpuBufferFlags bufferYFlags = GpuBufferFlags::frequentHostAccess
                              | GpuBufferFlags::large
                              | GpuBufferFlags::hostWriteAccess;

  GpuBuffer bufferY = gpu->allocateBuffer(bufferYSize, bufferYFlags);

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

  gpu::DenseLayer layer(*gpu, config, layerInputSize, miniBatchSize);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, weightsBuffer).WillByDefault(testing::Return(0));
  ON_CALL(nextLayer, deltaBuffer).WillByDefault(testing::Return(0));

  layer.allocateGpuResources(inputBuffer.handle, statusBuffer.handle, &nextLayer, bufferY.handle);

  layer.trainForward();
  gpu->flushQueue();

  Vector A(layerSize);
  gpu->retrieveBuffer(layer.test_activationsBuffer(), A.data());

  Vector expectedA = cpuDenseLayerTrainForward(config, W, B, inputs);

  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_NEAR(A[i], expectedA[i], FLOAT_TOLERANCE);
  }
}

void cpuDenseLayerBackprop(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs, const Matrix& nextW, const Vector& nextDelta, Vector& delta, Matrix& deltaW,
  Vector& deltaB) {

  testing::NiceMock<MockCpuLayer> nextLayer;

  ON_CALL(nextLayer, delta).WillByDefault(testing::ReturnRef(nextDelta.storage()));
  ON_CALL(nextLayer, W).WillByDefault(testing::ReturnRef(nextW));

  cpu::DenseLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());
  layer.updateDelta(inputs.storage(), nextLayer);

  delta = layer.delta();
  deltaW = layer.test_deltaW();
  deltaB = layer.test_deltaB();
}

TEST_F(GpuDenseLayerTest, backprop) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t miniBatchSize = 1;
  const size_t layerInputSize = 4;
  const size_t layerSize = 2;

  Vector Y({ 0.0, 1.0 });

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

  Vector inputs{ 0.5, 0.4, 0.3, 0.2 };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  Vector nextDelta({ 2, 3 });
  Matrix nextW({
    { 2, 5 },
    { 4, 1 }
  });

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

  gpu::DenseLayer layer(*gpu, config, layerInputSize, miniBatchSize);

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

  layer.allocateGpuResources(inputBuffer.handle, statusBuffer.handle, &nextLayer, bufferY.handle);

  layer.trainForward();
  layer.backprop();

  gpu->flushQueue();

  Vector delta(layerSize);
  Matrix deltaW(W.cols(), W.rows());
  Vector deltaB(B.size());

  gpu->retrieveBuffer(layer.deltaBuffer(), delta.data());
  gpu->retrieveBuffer(layer.test_deltaWBuffer(), deltaW.data());
  gpu->retrieveBuffer(layer.test_deltaBBuffer(), deltaB.data());

  Vector expectedDelta;
  Matrix expectedDeltaW;
  Vector expectedDeltaB;

  cpuDenseLayerBackprop(config, W, B, inputs, nextW, nextDelta, expectedDelta,
    expectedDeltaW, expectedDeltaB);

  for (size_t i = 0; i < delta.size(); ++i) {
    EXPECT_NEAR(delta[i], expectedDelta[i], FLOAT_TOLERANCE);
  }

  for (size_t j = 0; j < deltaW.rows(); ++j) {
    for (size_t i = 0; i < deltaW.cols(); ++i) {
      EXPECT_NEAR(deltaW.at(i, j), expectedDeltaW.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < deltaB.size(); ++i) {
    EXPECT_NEAR(deltaB[i], expectedDeltaB[i], FLOAT_TOLERANCE);
  }
}

TEST_F(GpuDenseLayerTest, updateParams) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t miniBatchSize = 1;
  const size_t layerInputSize = 4;
  const size_t layerSize = 2;

  Vector Y({ 0.0, 1.0 });

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

  gpu::DenseLayer layer(*gpu, config, layerInputSize, miniBatchSize);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  testing::NiceMock<MockGpuLayer> nextLayer;

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.allocateGpuResources(inputBuffer.handle, statusBuffer.handle, &nextLayer, bufferY.handle);

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

  for (size_t j = 0; j < expectedW.rows(); ++j) {
    for (size_t i = 0; i < expectedW.cols(); ++i) {
      EXPECT_NEAR(actualW.at(i, j), expectedW.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < expectedB.size(); ++i) {
    EXPECT_NEAR(actualB[i], expectedB[i], FLOAT_TOLERANCE);
  }
}
