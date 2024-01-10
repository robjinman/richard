#include "mock_logger.hpp"
#include "mock_gpu_layer.hpp"
#include "mock_cpu_layer.hpp"
#include <cpu/convolutional_layer.hpp>
#include <gpu/convolutional_layer.hpp>
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

class GpuConvolutionalLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

DataArray cpuConvolutionalLayerTrainForward(const nlohmann::json& config,
  const std::vector<cpu::ConvolutionalLayer::Filter>& filters, const Array3& inputs) {

  cpu::ConvolutionalLayer layer(config, inputs.shape());

  layer.test_setFilters(filters);

  layer.trainForward(inputs.storage());

  return layer.activations();
}

TEST_F(GpuConvolutionalLayerTest, trainForward) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  Array3 inputs({
    {
      { 0, 1, 2 },
      { 5, 6, 7 },
      { 8, 7, 6 },
    }, {
      { 5, 4, 3 },
      { 2, 1, 0 },
      { 1, 2, 5 }
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
  config["depth"] = 2;
  config["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  config["learnRate"] = 1.0;
  config["learnRateDecay"] = 1.0;
  config["dropoutRate"] = 0.0;

  gpu::ConvolutionalLayer layer(*gpu, config, { 3, 3, 2 }, true);

  cpu::ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({
    {
      { 5, 3 },
      { 1, 2 }
    }, {
      { 8, 4 },
      { 5, 3 }
    }
  });
  filter0.b = 7;

  cpu::ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({
    {
      { 5, 3 },
      { 1, 2 }
    }, {
      { 8, 4 },
      { 5, 3 }
    }
  });
  filter1.b = 3;

  DataArray kernelData = DataArray::concat({ filter0.K.storage(), filter1.K.storage() });
  Vector biasData{ filter0.b, filter1.b };

  layer.test_setKernels(kernelData);
  layer.test_setBiases(biasData.storage());

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, inputDeltaBuffer).WillByDefault(testing::Return(0));

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, &nextLayer, 0);

  layer.trainForward();
  gpu->flushQueue();

  Array3 A(layer.outputSize());
  gpu->retrieveBuffer(layer.test_activationsBuffer(), A.data());

  Array3 expectedA(cpuConvolutionalLayerTrainForward(config, { filter0, filter1 }, inputs),
    layer.outputSize());

  for (size_t k = 0; k < expectedA.D(); ++k) {
    for (size_t j = 0; j < expectedA.H(); ++j) {
      for (size_t i = 0; i < expectedA.W(); ++i) {
        EXPECT_NEAR(A.at(i, j, k), expectedA.at(i, j, k), FLOAT_TOLERANCE);
      }
    }
  }
}
/*
void cpuConvolutionalLayerBackprop(const nlohmann::json& config, const Matrix& W, const Vector& B,
  const Vector& inputs, const Vector& dA, Matrix& deltaW, Vector& deltaB) {

  cpu::ConvolutionalLayer layer(config, inputs.size());

  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());

  layer.trainForward(inputs.storage());
  layer.updateDeltas(inputs.storage(), dA.storage());

  deltaW = layer.test_deltaW();
  deltaB = layer.test_deltaB();
}

TEST_F(GpuConvolutionalLayerTest, backprop) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  size_t inputBufferSize = layerInputSize * sizeof(netfloat_t);

  GpuBufferFlags inputBufferFlags = GpuBufferFlags::large
                                  | GpuBufferFlags::hostWriteAccess;

  GpuBuffer inputBuffer = gpu->allocateBuffer(inputBufferSize, inputBufferFlags);

  Vector inputs{ 0.5, 0.4, 0.3, 0.2 };
  gpu->submitBufferData(inputBuffer.handle, inputs.data());

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  Vector nextDelta({ 2, 7 });
  Matrix nextW({
    { 2, 5 },
    { 4, 3 }
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

  gpu::ConvolutionalLayer layer(*gpu, config, layerInputSize, true);

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

  cpuConvolutionalLayerBackprop(config, W, B, inputs, dA, expectedDeltaW, expectedDeltaB);

  for (size_t j = 0; j < deltaW.rows(); ++j) {
    for (size_t i = 0; i < deltaW.cols(); ++i) {
      EXPECT_NEAR(deltaW.at(i, j), expectedDeltaW.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < deltaB.size(); ++i) {
    EXPECT_NEAR(deltaB[i], expectedDeltaB[i], FLOAT_TOLERANCE);
  }
}

TEST_F(GpuConvolutionalLayerTest, updateParams) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

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

  gpu::ConvolutionalLayer layer(*gpu, config, layerInputSize, true);

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