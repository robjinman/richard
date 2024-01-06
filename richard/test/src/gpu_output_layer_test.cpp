#include "mock_logger.hpp"
#include <cpu/output_layer.hpp>
#include <gpu/output_layer.hpp>
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

class GpuOutputLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

Vector cpuOutputLayerTrainForward(const nlohmann::json& config, const Matrix& W, const Vector& B,
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

  Vector inputs{ 0.5, 0.4, 0.3, 0.2 };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  nlohmann::json config;
  config["size"] = outputSize;
  config["learnRate"] = 0.1;
  config["learnRateDecay"] = 1.0;

  gpu::OutputLayer layer(*gpu, config, layerInputSize);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

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

void cpuOutputLayerBackprop(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs, const Vector& Y, Vector& delta, Matrix& deltaW, Vector& deltaB) {

  cpu::OutputLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());
  layer.updateDeltas(inputs.storage(), Y.storage());

  //delta = layer.delta();
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
  config["size"] = outputSize;
  config["learnRate"] = 0.1;
  config["learnRateDecay"] = 1.0;

  gpu::OutputLayer layer(*gpu, config, layerInputSize);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, nullptr, bufferY.handle);

  layer.trainForward();
  layer.backprop();

  gpu->flushQueue();

  Vector delta(outputSize);
  Matrix deltaW(W.cols(), W.rows());
  Vector deltaB(B.size());

  gpu->retrieveBuffer(layer.deltaBuffer(), delta.data());
  gpu->retrieveBuffer(layer.test_deltaWBuffer(), deltaW.data());
  gpu->retrieveBuffer(layer.test_deltaBBuffer(), deltaB.data());

  Vector expectedDelta;
  Matrix expectedDeltaW;
  Vector expectedDeltaB;
/*
  cpuOutputLayerBackprop(config, W, B, inputs, Y, expectedDelta, expectedDeltaW, expectedDeltaB);

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
  }*/
}

