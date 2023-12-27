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

class GpuOutputLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

Vector cpuOutputLayerTrainForward(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs) {

  cpu::OutputLayer layer(config, inputs.size());

  layer.setWeights(W.storage());
  layer.setBiases(B.storage());

  layer.trainForward(inputs.storage());

  return layer.activations();
}

TEST_F(GpuOutputLayerTest, trainForward) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  struct StatusBuffer {
    uint32_t epoch;
    netfloat_t cost;
    uint32_t sampleIndex;
  };

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t miniBatchSize = 4;
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
  status.cost = 0;

  nlohmann::json config;
  config["size"] = outputSize;
  config["learnRate"] = 0.1;
  config["learnRateDecay"] = 1.0;

  gpu::OutputLayer layer(*gpu, config, layerInputSize, miniBatchSize);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  layer.setWeights(W);
  layer.setBiases(B);

  layer.allocateGpuResources(inputBuffer.handle, statusBuffer.handle, nullptr, bufferY.handle);

  layer.trainForward();
  gpu->flushQueue();

  Vector A(outputSize);
  gpu->retrieveBuffer(layer.activationsBuffer(), A.data());

  Vector expectedA = cpuOutputLayerTrainForward(config, W, B, inputs);

  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_NEAR(A[i], expectedA[i], FLOAT_TOLERANCE);
  }
}
/*
TEST_F(GpuOutputLayerTest, backprop) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  struct StatusBuffer {
    uint32_t epoch;
    netfloat_t cost;
    uint32_t sampleIndex;
  };

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t miniBatchSize = 4;
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
  status.cost = 0;

  nlohmann::json config;
  config["size"] = outputSize;
  config["learnRate"] = 0.1;
  config["learnRateDecay"] = 1.0;

  gpu::OutputLayer layer(*gpu, config, layerInputSize, miniBatchSize);

  Matrix W({
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3, 0.2 }
  });

  Vector B({ 0.7, 0.8 });

  layer.setWeights(W);
  layer.setBiases(B);

  layer.allocateGpuResources(inputBuffer.handle, statusBuffer.handle, nullptr, bufferY.handle);

  layer.trainForward();
  gpu->flushQueue();

  Vector A(outputSize);
  gpu->retrieveBuffer(layer.activationsBuffer(), A.data());

  Vector expectedA = cpuOutputLayerTrainForward(config, W, B, inputs);

  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_NEAR(A[i], expectedA[i], FLOAT_TOLERANCE);
  }
}
*/
