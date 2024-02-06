#include "mock_logger.hpp"
#include <richard/utils.hpp>
#include <richard/file_system.hpp>
#include <richard/platform_paths.hpp>
#include <richard/cpu/dense_layer.hpp>
#include <richard/cpu/convolutional_layer.hpp>
#include <richard/cpu/max_pooling_layer.hpp>
#include <richard/cpu/output_layer.hpp>
#include <richard/gpu/dense_layer.hpp>
#include <richard/gpu/convolutional_layer.hpp>
#include <richard/gpu/max_pooling_layer.hpp>
#include <richard/gpu/output_layer.hpp>
#include <richard/gpu/gpu.hpp>
#include <gtest/gtest.h>

using namespace richard;

using richard::gpu::GpuPtr;
using richard::gpu::GpuBuffer;
using richard::gpu::GpuBufferFlags;

const double FLOAT_TOLERANCE = 0.000001;

const auto quadraticCost = [](const Vector& actual, const Vector& expected) {
  return (expected - actual).squareMagnitude() * 0.5f;
};

struct StatusBuffer {
  uint32_t epoch;
  uint32_t sampleIndex;
};

class GpuNeuralNetTest : public testing::Test {
  public:
    GpuNeuralNetTest()
      : m_fileSystem(createFileSystem()) {}

    virtual void SetUp() override {}
    virtual void TearDown() override {}

  protected:
    FileSystemPtr m_fileSystem;
};

void runCpuSimpleDenseNetwork(const Config& denseConfig, const Config& outputConfig,
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

    layer2.updateDeltas(layer1.activations(), y.storage());
    layer1.updateDeltas(x.storage(), layer2.inputDelta());

    layer1.updateParams(0);
    layer2.updateParams(0);
  }

  finalW1 = layer1.test_W();
  finalB1 = layer1.test_B();

  finalW2 = layer2.test_W();
  finalB2 = layer2.test_B();
}

TEST_F(GpuNeuralNetTest, simpleDenseNetwork) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  const size_t miniBatchSize = 1;
  const size_t layer1Size = 3;
  const size_t layer2Size = 2;
  const size_t inputSize = 4;

  std::vector<Vector> X{
    Vector{ 0.6f, 0.2f, 0.5f, 0.7f },
    Vector{ 0.7f, 0.1f, 0.9f, 0.5f }
  };
  std::vector<Vector> Y{
    Vector{ 1.f, 0.f },
    Vector{ 0.f, 1.f }
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

  Config layer1Config;
  layer1Config.setNumber("size", layer1Size);
  layer1Config.setNumber("learnRate", 0.1);
  layer1Config.setNumber("learnRateDecay", 1.0);
  layer1Config.setNumber("dropoutRate", 0.0);

  Config layer2Config;
  layer2Config.setNumber("size", layer2Size);
  layer2Config.setNumber("learnRate", 0.1);
  layer2Config.setNumber("learnRateDecay", 1.0);

  gpu::DenseLayer layer1(*gpu, *fileSystem, *platformPaths, layer1Config, inputSize, true);
  gpu::OutputLayer layer2(*gpu, *fileSystem, *platformPaths, layer2Config, layer1Size);

  Matrix W1({
    { 0.7f, 0.3f, 0.1f, 0.4f },
    { 0.2f, 0.9f, 0.4f, 0.5f },
    { 0.1f, 0.6f, 0.2f, 0.7f },
  });

  Vector B1({ 0.5f, 0.4f, 0.2f });

  Matrix W2({
    { 0.8f, 0.3f, 0.1f },
    { 0.9f, 0.4f, 0.5f }
  });

  Vector B2({ 0.4f, 0.2f });

  layer1.test_setWeights(W1.storage());
  layer1.test_setBiases(B1.storage());

  layer2.test_setWeights(W2.storage());
  layer2.test_setBiases(B2.storage());

  layer1.allocateGpuBuffers();
  layer2.allocateGpuBuffers();

  layer1.createGpuShaders(bufferX.handle, statusBuffer.handle, &layer2, bufferY.handle);
  layer2.createGpuShaders(layer1.outputBuffer(), statusBuffer.handle, nullptr, bufferY.handle);

  GpuBufferFlags costsBufferFlags = GpuBufferFlags::frequentHostAccess
                                  | GpuBufferFlags::large
                                  | GpuBufferFlags::hostReadAccess;

  GpuBuffer costsBuffer = gpu->allocateBuffer(layer2Size * sizeof(netfloat_t), costsBufferFlags);

  gpu::GpuBufferBindings computeCostsBuffers{
    { statusBuffer.handle, gpu::BufferAccessMode::write },
    { layer2.outputBuffer(), gpu::BufferAccessMode::read },
    { bufferY.handle, gpu::BufferAccessMode::read },
    { costsBuffer.handle, gpu::BufferAccessMode::write }
  };

  gpu::SpecializationConstants computeCostsConstants{
    { gpu::SpecializationConstant::Type::uint_type, static_cast<uint32_t>(miniBatchSize) }
  };

  auto computeCostsShaderPath = platformPaths->get("shaders", "compute_costs.spv");
  auto computeCostsShaderCode = fileSystem->loadBinaryFile(computeCostsShaderPath);

  gpu::ShaderHandle computeCostsShader = gpu->addShader("compute_costs.spv", computeCostsShaderCode,
    computeCostsBuffers, computeCostsConstants, { static_cast<uint32_t>(layer2Size), 1, 1 });

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

  runCpuSimpleDenseNetwork(layer1Config, layer2Config, W1, B1, W2, B2, X, Y, expectedW1, expectedB1,
    expectedW2, expectedB2);

  EXPECT_EQ(expectedW1.cols(), finalW1.cols());
  EXPECT_EQ(expectedW1.rows(), finalW1.rows());

  for (size_t j = 0; j < expectedW1.rows(); ++j) {
    for (size_t i = 0; i < expectedW1.cols(); ++i) {
      EXPECT_NEAR(finalW1.at(i, j), expectedW1.at(i, j), FLOAT_TOLERANCE);
    }
  }

  EXPECT_EQ(expectedB1.size(), finalB1.size());

  for (size_t i = 0; i < expectedB1.size(); ++i) {
    EXPECT_NEAR(finalB1[i], expectedB1[i], FLOAT_TOLERANCE);
  }

  EXPECT_EQ(expectedW2.cols(), finalW2.cols());
  EXPECT_EQ(expectedW2.rows(), finalW2.rows());

  for (size_t j = 0; j < expectedW2.rows(); ++j) {
    for (size_t i = 0; i < expectedW2.cols(); ++i) {
      EXPECT_NEAR(finalW2.at(i, j), expectedW2.at(i, j), FLOAT_TOLERANCE);
    }
  }

  EXPECT_EQ(expectedB2.size(), finalB2.size());

  for (size_t i = 0; i < expectedB2.size(); ++i) {
    EXPECT_NEAR(finalB2[i], expectedB2[i], FLOAT_TOLERANCE);
  }
}

netfloat_t runCpuSimpleConvNetwork(const Config& config1, const Config& config2,
  const Config& config3, const std::vector<cpu::ConvolutionalLayer::Filter>& filters,
  const Matrix& W2, const Vector& B2, const std::vector<Array3>& X, const std::vector<Vector>& Y,
  std::vector<Kernel>& finalK, Vector& finalB1, Matrix& finalW2, Vector& finalB2) {

  cpu::ConvolutionalLayer layer1(config1, { 3, 3, 2 });
  cpu::MaxPoolingLayer layer2(config2, { 2, 2, 2 });
  cpu::OutputLayer layer3(config3, calcProduct(layer2.outputSize()));

  layer1.test_setFilters(filters);

  layer3.test_setWeights(W2.storage());
  layer3.test_setBiases(B2.storage());

  netfloat_t cost = 0.0;

  for (size_t i = 0; i < X.size(); ++i) {
    const Array3& x = X[i];
    const Vector& y = Y[i];

    layer1.trainForward(x.storage());
    layer2.trainForward(layer1.activations());
    layer3.trainForward(layer2.activations());

    cost += quadraticCost(layer3.activations(), y);

    layer3.updateDeltas(layer2.activations(), y.storage());
    layer2.updateDeltas(layer1.activations(), layer3.inputDelta());
    layer1.updateDeltas(x.storage(), layer2.inputDelta());

    layer1.updateParams(0);
    layer2.updateParams(0);
    layer3.updateParams(0);
  }

  cost /= X.size();

  const auto& finalFilters = layer1.test_filters();

  finalB1 = Vector(finalFilters.size());

  for (size_t i = 0; i < finalFilters.size(); ++i) {
    finalK.push_back(finalFilters[i].K);
    finalB1[i] = finalFilters[i].b;
  }

  finalW2 = layer3.test_W();
  finalB2 = layer3.test_B();

  return cost;
}

TEST_F(GpuNeuralNetTest, simpleConvNetwork) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  const size_t miniBatchSize = 1;
  const size_t outputLayerSize = 2;

  std::vector<Array3> X{
    Array3{{
      { 0.7f, 0.1f, 0.3f },
      { 0.8f, 0.6f, 0.2f },
      { 0.2f, 0.9f, 0.5f }
    }, {
      { 0.8f, 0.5f, 0.4f },
      { 0.9f, 0.1f, 0.2f },
      { 0.5f, 0.8f, 0.6f }
    }},
    Array3{{
      { 0.1f, 0.8f, 0.6f },
      { 0.5f, 0.4f, 0.1f },
      { 0.2f, 0.5f, 0.7f }
    }, {
      { 0.3f, 0.5f, 0.2f },
      { 0.8f, 0.1f, 0.6f },
      { 0.1f, 0.8f, 0.4f }
    }},
    Array3{{
      { 0.2f, 0.3f, 0.5f },
      { 0.4f, 0.8f, 0.2f },
      { 0.1f, 0.7f, 0.2f }
    }, {
      { 0.9f, 0.2f, 0.1f },
      { 0.2f, 0.3f, 0.1f },
      { 0.4f, 0.6f, 0.5f }
    }}
  };
  std::vector<Vector> Y{
    Vector{ 1.f, 0.f },
    Vector{ 0.f, 1.f },
    Vector{ 1.f, 0.f }
  };

  Size3 inputShape{ 3, 3, 2 };
  size_t inputSize = calcProduct(inputShape);

  size_t bufferXSize = miniBatchSize * inputSize * sizeof(netfloat_t);
  size_t bufferYSize = miniBatchSize * outputLayerSize * sizeof(netfloat_t);

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

  Config layer1Config;
  layer1Config.setNumber("depth", 2);
  layer1Config.setNumberArray<size_t>("kernelSize", { 2, 2 });
  layer1Config.setNumber("learnRate", 0.1);
  layer1Config.setNumber("learnRateDecay", 1.0);
  layer1Config.setNumber("dropoutRate", 0.0);

  Config layer2Config;
  layer2Config.setNumberArray<size_t>("regionSize", { 2, 2 });

  Config layer3Config;
  layer3Config.setNumber("size", 2);
  layer3Config.setNumber("learnRate", 0.1);
  layer3Config.setNumber("learnRateDecay", 1.0);

  gpu::ConvolutionalLayer layer1(*gpu, *fileSystem, *platformPaths, layer1Config, { 3, 3, 2 },
    true);
  gpu::MaxPoolingLayer layer2(*gpu, *fileSystem, *platformPaths, layer2Config, { 2, 2, 2 });
  gpu::OutputLayer layer3(*gpu, *fileSystem, *platformPaths, layer3Config, 2);

  cpu::ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({
    {
      { 0.5f, 0.3f },
      { 0.1f, 0.2f }
    }, {
      { 0.8f, 0.4f },
      { 0.5f, 0.3f }
    }
  });
  filter0.b = 0.7f;

  cpu::ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({
    {
      { 0.2f, 0.4f },
      { 0.5f, 0.6f }
    }, {
      { 0.4f, 0.1f },
      { 0.2f, 0.9f }
    }
  });
  filter1.b = 0.3f;

  size_t kernelSize = 2 * 2 * 2;

  DataArray KData = DataArray::concat({ filter0.K.storage(), filter1.K.storage() });
  Vector biasData{ filter0.b, filter1.b };

  layer1.test_setKernels(KData);
  layer1.test_setBiases(biasData.storage());

  Matrix W2({
    { 0.8f, 0.3f },
    { 0.9f, 0.4f }
  });

  Vector B2({ 0.4f, 0.2f });

  layer3.test_setWeights(W2.storage());
  layer3.test_setBiases(B2.storage());

  layer1.allocateGpuBuffers();
  layer2.allocateGpuBuffers();
  layer3.allocateGpuBuffers();

  layer1.createGpuShaders(bufferX.handle, statusBuffer.handle, &layer2, bufferY.handle);
  layer2.createGpuShaders(layer1.outputBuffer(), statusBuffer.handle, &layer3, bufferY.handle);
  layer3.createGpuShaders(layer2.outputBuffer(), statusBuffer.handle, nullptr, bufferY.handle);

  GpuBufferFlags costsBufferFlags = GpuBufferFlags::frequentHostAccess
                                  | GpuBufferFlags::large
                                  | GpuBufferFlags::hostReadAccess;

  GpuBuffer costsBuffer = gpu->allocateBuffer(outputLayerSize * sizeof(netfloat_t),
    costsBufferFlags);

  gpu::GpuBufferBindings computeCostsBuffers{
    { statusBuffer.handle, gpu::BufferAccessMode::write },
    { layer3.outputBuffer(), gpu::BufferAccessMode::read },
    { bufferY.handle, gpu::BufferAccessMode::read },
    { costsBuffer.handle, gpu::BufferAccessMode::write }
  };

  gpu::SpecializationConstants computeCostsConstants{
    { gpu::SpecializationConstant::Type::uint_type, static_cast<uint32_t>(miniBatchSize) }
  };

  auto computeCostsShaderPath = platformPaths->get("shaders", "compute_costs.spv");
  auto computeCostsShaderCode = fileSystem->loadBinaryFile(computeCostsShaderPath);

  gpu::ShaderHandle computeCostsShader = gpu->addShader("compute_costs.spv", computeCostsShaderCode,
    computeCostsBuffers, computeCostsConstants, { static_cast<uint32_t>(outputLayerSize), 1, 1 });

  for (size_t i = 0; i < X.size(); ++i) {
    memcpy(bufferX.data, X[i].data(), inputSize * sizeof(netfloat_t));
    memcpy(bufferY.data, Y[i].data(), outputLayerSize * sizeof(netfloat_t));

    layer1.trainForward();
    layer2.trainForward();
    layer3.trainForward();

    layer3.backprop();
    layer2.backprop();
    layer1.backprop();

    gpu->queueShader(computeCostsShader);

    layer1.updateParams();
    layer2.updateParams();
    layer3.updateParams();

    gpu->flushQueue();
  }

  layer1.retrieveBuffers();
  layer2.retrieveBuffers();
  layer3.retrieveBuffers();

  const DataArray& actualK = layer1.test_kernels();
  const Vector& actualB1 = layer1.test_biases();

  const Matrix& actualW2 = layer3.test_W();
  const Vector& actualB2 = layer3.test_B();

  netfloat_t actualCost = 0.0;
  for (size_t i = 0; i < outputLayerSize; ++i) {
    actualCost += reinterpret_cast<const netfloat_t*>(costsBuffer.data)[i];
  }
  actualCost /= X.size();

  std::vector<Kernel> expectedK;
  Vector expectedB1;
  Matrix expectedW2;
  Vector expectedB2;

  netfloat_t expectedCost = runCpuSimpleConvNetwork(layer1Config, layer2Config, layer3Config,
    { filter0, filter1 }, W2, B2, X, Y, expectedK, expectedB1, expectedW2, expectedB2);

  EXPECT_NEAR(actualCost, expectedCost, FLOAT_TOLERANCE);

  for (size_t d = 0; d < expectedK.size(); ++d) {
    ConstKernelPtr pK = Kernel::createShallow(actualK.data() + d * kernelSize, 2, 2, 2);
    const Kernel& K = *pK;

    for (size_t k = 0; k < K.D(); ++k) {
      for (size_t j = 0; j < K.H(); ++j) {
        for (size_t i = 0; i < K.W(); ++i) {
          EXPECT_NEAR(K.at(i, j, k), expectedK[d].at(i, j, k), FLOAT_TOLERANCE);
        }
      }
    }

    EXPECT_NEAR(actualB1[d], expectedB1[d], FLOAT_TOLERANCE);
  }

  EXPECT_EQ(expectedW2.cols(), actualW2.cols());
  EXPECT_EQ(expectedW2.rows(), actualW2.rows());

  for (size_t j = 0; j < expectedW2.rows(); ++j) {
    for (size_t i = 0; i < expectedW2.cols(); ++i) {
      EXPECT_NEAR(actualW2.at(i, j), expectedW2.at(i, j), FLOAT_TOLERANCE);
    }
  }

  EXPECT_EQ(actualB2.size(), expectedB2.size());

  for (size_t i = 0; i < expectedB2.size(); ++i) {
    EXPECT_NEAR(actualB2[i], expectedB2[i], FLOAT_TOLERANCE);
  }
}
