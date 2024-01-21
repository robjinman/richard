#include "mock_logger.hpp"
#include "mock_gpu_layer.hpp"
#include "mock_cpu_layer.hpp"
#include <cpu/convolutional_layer.hpp>
#include <gpu/convolutional_layer.hpp>
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

class GpuConvolutionalLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

DataArray cpuConvolutionalLayerTrainForward(const Config& config,
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

  Config config;
  config.setInteger("depth", 2);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", 1.0);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::ConvolutionalLayer layer(*gpu, *fileSystem, *platformPaths, config, { 3, 3, 2 }, true);

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
      { 2, 4 },
      { 5, 6 }
    }, {
      { 4, 1 },
      { 2, 9 }
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
  gpu->retrieveBuffer(layer.outputBuffer(), A.data());

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

void cpuConvolutionalLayerBackprop(const Config& config,
  const std::vector<cpu::ConvolutionalLayer::Filter>& filters, const Array3& inputs,
  const Vector& dA, std::vector<Kernel>& deltaK, Vector& deltaB) {

  deltaB = Vector(filters.size());

  cpu::ConvolutionalLayer layer(config, inputs.shape());

  layer.test_setFilters(filters);

  layer.trainForward(inputs.storage());
  layer.updateDeltas(inputs.storage(), dA.storage());

  const auto& deltas = layer.test_filterDeltas();

  for (size_t i = 0; i < deltas.size(); ++i) {
    deltaK.push_back(deltas[i].K);
    deltaB[i] = deltas[i].b;
  }
}

TEST_F(GpuConvolutionalLayerTest, backprop) {
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

  Array3 dA({
    {
      { 2, 1 },
      { 4, 7 }
    }, {
      { 5, 2 },
      { 9, 8 }
    }
  });

  GpuBufferFlags bufferFlags = GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess;
  GpuBuffer bufferDeltaA = gpu->allocateBuffer(dA.size() * sizeof(netfloat_t), bufferFlags);

  gpu->submitBufferData(bufferDeltaA.handle, dA.data());

  size_t layerDepth = 2;

  Config config;
  config.setInteger("depth", layerDepth);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", 1.0);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::ConvolutionalLayer layer(*gpu, *fileSystem, *platformPaths, config, { 3, 3, 2 }, true);

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
      { 2, 4 },
      { 5, 6 }
    }, {
      { 4, 1 },
      { 2, 9 }
    }
  });
  filter1.b = 3;

  DataArray kernelData = DataArray::concat({ filter0.K.storage(), filter1.K.storage() });
  Vector biasData{ filter0.b, filter1.b };

  layer.test_setKernels(kernelData);
  layer.test_setBiases(biasData.storage());

  testing::NiceMock<MockGpuLayer> nextLayer;
  ON_CALL(nextLayer, inputDeltaBuffer).WillByDefault(testing::Return(bufferDeltaA.handle));

  layer.allocateGpuBuffers();
  layer.createGpuShaders(inputBuffer.handle, statusBuffer.handle, &nextLayer, 0);

  layer.trainForward();
  layer.backprop();

  gpu->flushQueue();

  size_t kernelSize = filter0.K.size();

  DataArray deltaKData(kernelSize * layerDepth);
  Vector deltaB(layerDepth);

  gpu->retrieveBuffer(layer.test_deltaKBuffer(), deltaKData.data());
  gpu->retrieveBuffer(layer.test_deltaBBuffer(), deltaB.data());

  std::vector<Kernel> expectedDeltaK;
  Vector expectedDeltaB;

  cpuConvolutionalLayerBackprop(config, { filter0, filter1 }, inputs, dA.storage(), expectedDeltaK,
    expectedDeltaB);

  for (size_t d = 0; d < expectedDeltaK.size(); ++d) {
    ConstKernelPtr pDeltaK = Kernel::createShallow(deltaKData.data() + d * kernelSize, 2, 2, 2);
    const Kernel& deltaK = *pDeltaK;

    for (size_t k = 0; k < deltaK.D(); ++k) {
      for (size_t j = 0; j < deltaK.H(); ++j) {
        for (size_t i = 0; i < deltaK.W(); ++i) {
          EXPECT_NEAR(deltaK.at(i, j, k), expectedDeltaK[d].at(i, j, k), FLOAT_TOLERANCE);
        }
      }
    }

    EXPECT_NEAR(deltaB[d], expectedDeltaB[d], FLOAT_TOLERANCE);
  }
}

TEST_F(GpuConvolutionalLayerTest, updateParams) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = gpu::createGpu(logger);

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.epoch = 0;
  status.sampleIndex = 0;

  size_t layerDepth = 2;
  netfloat_t learnRate = 0.47;

  Config config;
  config.setInteger("depth", layerDepth);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", learnRate);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  FileSystemPtr fileSystem = createFileSystem();
  PlatformPathsPtr platformPaths = createPlatformPaths();

  gpu::ConvolutionalLayer layer(*gpu, *fileSystem, *platformPaths, config, { 3, 3, 2 }, true);

  Kernel K1{
    {
      { 0.5, 0.3 },
      { 0.1, 0.2 }
    }, {
      { 0.8, 0.4 },
      { 0.5, 0.3 }
    }
  };

  Kernel K2{
    {
      { 0.2, 0.4 },
      { 0.5, 0.6 }
    }, {
      { 0.4, 0.1 },
      { 0.2, 0.9 }
    }
  };

  Vector B{ 9, 5 };
  DataArray K = DataArray::concat({ K1.storage(), K2.storage() });

  layer.test_setKernels(K);
  layer.test_setBiases(B.storage());

  testing::NiceMock<MockGpuLayer> nextLayer;

  layer.allocateGpuBuffers();
  layer.createGpuShaders(0, statusBuffer.handle, &nextLayer, 0);

  Kernel deltaK1{
    {
      { 0.1, 0.2 },
      { 0.6, 0.4 }
    }, {
      { 0.8, 0.7 },
      { 0.5, 0.8 }
    }
  };

  Kernel deltaK2{
    {
      { 0.1, 0.5 },
      { 0.9, 0.7 }
    }, {
      { 0.3, 0.2 },
      { 0.3, 0.6 }
    }
  };

  DataArray deltaK = DataArray::concat({ deltaK1.storage(), deltaK2.storage() });
  Vector deltaB{ 6, 2 };

  gpu->submitBufferData(layer.test_deltaKBuffer(), deltaK.data());
  gpu->submitBufferData(layer.test_deltaBBuffer(), deltaB.data());

  layer.updateParams();

  gpu->flushQueue();

  layer.retrieveBuffers();

  Kernel expectedK1 = K1 - deltaK1 * learnRate;
  Kernel expectedK2 = K2 - deltaK2 * learnRate;
  Vector expectedB = B - deltaB * learnRate;

  layer.retrieveBuffers();

  const DataArray& actualK = layer.test_kernels();
  ConstKernelPtr pActualK1 = Kernel::createShallow(actualK.data(), K1.shape());
  const Kernel& actualK1 = *pActualK1;
  ConstKernelPtr pActualK2 = Kernel::createShallow(actualK.data() + K1.size(), K2.shape());
  const Kernel& actualK2 = *pActualK2;
  const Vector& actualB = layer.test_biases();

  EXPECT_EQ(expectedK1.shape(), actualK1.shape());
  EXPECT_EQ(expectedK2.shape(), actualK2.shape());

  for (size_t k = 0; k < expectedK1.D(); ++k) {
    for (size_t j = 0; j < expectedK1.H(); ++j) {
      for (size_t i = 0; i < expectedK1.W(); ++i) {
        EXPECT_NEAR(expectedK1.at(i, j, k), actualK1.at(i, j, k), FLOAT_TOLERANCE);
      }
    }
  }

  for (size_t k = 0; k < expectedK2.D(); ++k) {
    for (size_t j = 0; j < expectedK2.H(); ++j) {
      for (size_t i = 0; i < expectedK2.W(); ++i) {
        EXPECT_NEAR(expectedK2.at(i, j, k), actualK2.at(i, j, k), FLOAT_TOLERANCE);
      }
    }
  }

  EXPECT_EQ(expectedB.size(), actualB.size());

  for (size_t i = 0; i < expectedB.size(); ++i) {
    EXPECT_NEAR(actualB[i], expectedB[i], FLOAT_TOLERANCE);
  }
}
