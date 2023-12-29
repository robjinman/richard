#include "mock_logger.hpp"
#include <util.hpp>
#include <cpu/dense_layer.hpp>
#include <cpu/output_layer.hpp>
#include <gpu/dense_layer.hpp>
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

class GpuNeuralNetTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

void runCpuNetwork(const nlohmann::json& denseConfig, const nlohmann::json& outputConfig,
  const Matrix& W1, const Vector& B1, const Matrix& W2, const Vector& B2,
  const std::vector<Vector>& X, const std::vector<Vector>& Y, Matrix& finalW1, Vector& finalB1,
  Matrix& finalW2, Vector& finalB2) {

  size_t inputSize = X[0].size();

  cpu::DenseLayer layer1(denseConfig, inputSize);
  cpu::OutputLayer layer2(outputConfig, calcProduct(layer1.outputSize()));

  layer1.test_setWeights(W1.storage());
  layer1.test_setBiases(B1.storage());

  layer2.test_setWeights(W2.storage());
  layer2.test_setBiases(B2.storage());

  for (size_t i = 0; i < X.size(); ++i) {
    const Vector& x = X[i];
    const Vector& y = Y[i];

    layer1.trainForward(x.storage());
    layer2.trainForward(layer1.activations());

    layer2.updateDelta(layer1.activations(), y.storage());
    layer1.updateDelta(x.storage(), layer2);

    layer1.updateParams(0);
    layer2.updateParams(0);
  }

  finalW1 = layer1.W();
  finalB1 = layer1.test_B();

  finalW2 = layer2.W();
  finalB2 = layer2.test_B();
}

TEST_F(GpuNeuralNetTest, simpleNetwork) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  const size_t miniBatchSize = 1;
  const size_t layer1Size = 3;
  const size_t layer2Size = 2;
  const size_t inputSize = 4;

  std::vector<Vector> X{
    Vector{ 0.6, 0.2, 0.5, 0.7 },
    Vector{ 0.7, 0.1, 0.9, 0.5 }
  };
  std::vector<Vector> Y{
    Vector{ 1.0, 0.0 },
    Vector{ 0.0, 1.0 }
  };

  size_t bufferXSize = miniBatchSize * inputSize * sizeof(netfloat_t);
  size_t bufferYSize = miniBatchSize * layer2Size * sizeof(netfloat_t);

  GpuBufferFlags bufferXFlags = GpuBufferFlags::frequentHostAccess
                              | GpuBufferFlags::large
                              | GpuBufferFlags::hostWriteAccess;

  GpuBuffer bufferX = gpu->allocateBuffer(bufferXSize, bufferXFlags);

  GpuBufferFlags bufferYFlags = GpuBufferFlags::frequentHostAccess
                              | GpuBufferFlags::large
                              | GpuBufferFlags::hostWriteAccess;

  GpuBuffer bufferY = gpu->allocateBuffer(bufferYSize, bufferYFlags);

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  nlohmann::json layer1Config;
  layer1Config["size"] = layer1Size;
  layer1Config["learnRate"] = 0.1;
  layer1Config["learnRateDecay"] = 1.0;
  layer1Config["dropoutRate"] = 0.0;

  nlohmann::json layer2Config;
  layer2Config["size"] = layer2Size;
  layer2Config["learnRate"] = 0.1;
  layer2Config["learnRateDecay"] = 1.0;

  gpu::DenseLayer layer1(*gpu, layer1Config, inputSize, true);
  gpu::OutputLayer layer2(*gpu, layer2Config, layer1Size);

  Matrix W1({
    { 0.7, 0.3, 0.1, 0.4 },
    { 0.2, 0.9, 0.4, 0.5 },
    { 0.1, 0.6, 0.2, 0.7 },
  });

  Vector B1({ 0.5, 0.4, 0.2 });

  Matrix W2({
    { 0.8, 0.3, 0.1 },
    { 0.9, 0.4, 0.5 }
  });

  Vector B2({ 0.4, 0.2 });

  layer1.test_setWeights(W1.storage());
  layer1.test_setBiases(B1.storage());

  layer2.test_setWeights(W2.storage());
  layer2.test_setBiases(B2.storage());

  layer1.allocateGpuBuffers();
  layer2.allocateGpuBuffers();

  layer1.createGpuShaders(bufferX.handle, statusBuffer.handle, &layer2, bufferY.handle);
  layer2.createGpuShaders(layer1.outputBuffer(), statusBuffer.handle, nullptr, bufferY.handle);

  const std::string shaderIncludesDir = "./shaders";
  const std::string computeCostsSrc = loadFile("./shaders/compute_costs.glsl");

  GpuBufferFlags costsBufferFlags = GpuBufferFlags::frequentHostAccess
                                  | GpuBufferFlags::large
                                  | GpuBufferFlags::hostReadAccess;

  GpuBuffer costsBuffer = gpu->allocateBuffer(layer2Size * sizeof(netfloat_t), costsBufferFlags);

  gpu::GpuBufferBindings computeCostsBuffers{
    statusBuffer.handle,
    layer2.outputBuffer(),
    bufferY.handle,
    costsBuffer.handle
  };

  gpu::SpecializationConstants computeCostsConstants{
    { gpu::SpecializationConstant::Type::uint_type, static_cast<uint32_t>(miniBatchSize) }
  };

  gpu::ShaderHandle computeCostsShader = gpu->compileShader(computeCostsSrc, computeCostsBuffers,
    computeCostsConstants, { static_cast<uint32_t>(layer2Size), 1, 1 }, shaderIncludesDir);

  for (size_t i = 0; i < X.size(); ++i) {
    memcpy(bufferX.data, X[i].data(), inputSize * sizeof(netfloat_t));
    memcpy(bufferY.data, Y[i].data(), layer2Size * sizeof(netfloat_t));

    layer1.trainForward();
    layer2.trainForward();

    layer2.backprop();
    layer1.backprop();

    gpu->queueShader(computeCostsShader);

    layer1.updateParams();
    layer2.updateParams();

    gpu->flushQueue();
  }

  layer1.retrieveBuffers();
  layer2.retrieveBuffers();

  const Matrix& finalW1 = layer1.test_W();
  const Vector& finalB1 = layer1.test_B();

  const Matrix& finalW2 = layer2.test_W();
  const Vector& finalB2 = layer2.test_B();

  Matrix expectedW1;
  Vector expectedB1;
  Matrix expectedW2;
  Vector expectedB2;

  runCpuNetwork(layer1Config, layer2Config, W1, B1, W2, B2, X, Y, expectedW1, expectedB1,
    expectedW2, expectedB2);

  for (size_t j = 0; j < expectedW1.rows(); ++j) {
    for (size_t i = 0; i < expectedW1.cols(); ++i) {
      EXPECT_NEAR(finalW1.at(i, j), expectedW1.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < expectedB1.size(); ++i) {
    EXPECT_NEAR(finalB1[i], expectedB1[i], FLOAT_TOLERANCE);
  }

  for (size_t j = 0; j < expectedW2.rows(); ++j) {
    for (size_t i = 0; i < expectedW2.cols(); ++i) {
      EXPECT_NEAR(finalW2.at(i, j), expectedW2.at(i, j), FLOAT_TOLERANCE);
    }
  }

  for (size_t i = 0; i < expectedB2.size(); ++i) {
    EXPECT_NEAR(finalB2[i], expectedB2[i], FLOAT_TOLERANCE);
  }
}
