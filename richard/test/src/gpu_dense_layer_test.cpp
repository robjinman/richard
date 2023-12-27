#include "mock_logger.hpp"
#include "mock_gpu_layer.hpp"
#include <cpu/dense_layer.hpp>
#include <gpu/dense_layer.hpp>
#include <gpu/gpu.hpp>
#include <gtest/gtest.h>

using namespace richard;

using richard::gpu::GpuPtr;
using richard::gpu::GpuBuffer;
using richard::gpu::GpuBufferFlags;

const double FLOAT_TOLERANCE = 0.0001;

class GpuDenseLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

Vector cpuDenseLayerTrainForward(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs) {

  cpu::DenseLayer layer(config, inputs.size());

  layer.setWeights(W.storage());
  layer.setBiases(B.storage());

  layer.trainForward(inputs.storage());

  return layer.activations();
}

TEST_F(GpuDenseLayerTest, trainForward) {
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
  status.cost = 0;

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

  layer.setWeights(W);
  layer.setBiases(B);

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, weightsBuffer).WillByDefault(testing::Return(0));
  ON_CALL(nextLayer, deltaBuffer).WillByDefault(testing::Return(0));

  layer.allocateGpuResources(inputBuffer.handle, statusBuffer.handle, &nextLayer, bufferY.handle);

  layer.trainForward();
  gpu->flushQueue();

  Vector A(layerSize);
  gpu->retrieveBuffer(layer.activationsBuffer(), A.data());

  Vector expectedA = cpuDenseLayerTrainForward(config, W, B, inputs);

  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_NEAR(A[i], expectedA[i], FLOAT_TOLERANCE);
  }
}
