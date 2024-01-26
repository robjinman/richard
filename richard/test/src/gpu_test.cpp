#include "mock_logger.hpp"
#include <types.hpp>
#include <file_system.hpp>
#include <math.hpp>
#include <gpu/gpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>

using namespace richard;
using namespace richard::gpu;

const double FLOAT_TOLERANCE = 0.0001;

class GpuTest : public testing::Test {
  public:
    GpuTest()
      : m_fileSystem(createFileSystem()) {}

    virtual void SetUp() override {}
    virtual void TearDown() override {}

  protected:
    FileSystemPtr m_fileSystem;
};

TEST_F(GpuTest, constructionAndDestruction) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);
}

TEST_F(GpuTest, bufferSubmitAndRetrieve) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);

  std::array<netfloat_t, 16> data;
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<netfloat_t>(i);
  }

  GpuBuffer buffer = gpu->allocateBuffer(data.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  gpu->submitBufferData(buffer.handle, data.data());

  std::array<netfloat_t, 16> data2;
  gpu->retrieveBuffer(buffer.handle, data2.data());

  EXPECT_EQ(buffer.data, nullptr);
  EXPECT_EQ(data, data2);
}

TEST_F(GpuTest, runShader) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);

  const size_t bufferSize = 16;

  std::array<netfloat_t, bufferSize> data{};

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<netfloat_t>(i);
  }

  GpuBuffer buffer = gpu->allocateBuffer(data.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  gpu->submitBufferData(buffer.handle, data.data());

  auto shaderCode = m_fileSystem->loadBinaryFile("test/shaders/simple_shader.spv");

  GpuBufferBindings buffers{
    { buffer.handle, BufferAccessMode::write }
  };

  ShaderHandle shader = gpu->addShader("simple_shader", shaderCode, buffers, {},
    { bufferSize, 1, 1 });

  gpu->queueShader(shader);
  gpu->flushQueue();

  std::array<netfloat_t, bufferSize> expected{};
  std::transform(data.begin(), data.end(), expected.begin(), [](netfloat_t x) { return x * 2.f; });

  gpu->retrieveBuffer(buffer.handle, data.data());

  EXPECT_EQ(data, expected);
}

TEST_F(GpuTest, structuredBuffer) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);

  struct StatusBuffer {
    uint32_t epoch;
    netfloat_t cost;
    uint32_t sampleIndex;
  };

  GpuBufferFlags statusBufferFlags = GpuBufferFlags::frequentHostAccess
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;
  GpuBuffer statusBuffer = gpu->allocateBuffer(sizeof(StatusBuffer), statusBufferFlags);

  StatusBuffer& status = *reinterpret_cast<StatusBuffer*>(statusBuffer.data);
  status.cost = 100.0;

  auto shaderCode = m_fileSystem->loadBinaryFile("test/shaders/structured_buffer.spv");

  GpuBufferBindings buffers{
    { statusBuffer.handle, BufferAccessMode::write }
  };

  ShaderHandle shader = gpu->addShader("structured_buffer", shaderCode, buffers, {},
    { 16, 1, 1 });

  gpu->queueShader(shader);
  gpu->flushQueue();

  EXPECT_NEAR(status.cost, 223.45, FLOAT_TOLERANCE);
}

TEST_F(GpuTest, matrixMultiply) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);

  Matrix M({
    { 4, 2, 5, 7, 4 },
    { 7, 3, 8, 2, 8 },
    { 9, 1, 6, 1, 2 },
    { 6, 3, 4, 8, 9 },
  });

  Vector V({ 5, 3, 9, 8, 2 });
  Vector R(M.rows());

  GpuBuffer bufferM = gpu->allocateBuffer(M.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  GpuBuffer bufferV = gpu->allocateBuffer(V.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  GpuBuffer bufferR = gpu->allocateBuffer(R.size() * sizeof(netfloat_t), GpuBufferFlags::large);

  gpu->submitBufferData(bufferM.handle, M.data());
  gpu->submitBufferData(bufferV.handle, V.data());

  auto shaderCode = m_fileSystem->loadBinaryFile("test/shaders/matrix_multiply.spv");

  GpuBufferBindings buffers{
    { bufferM.handle, BufferAccessMode::read },
    { bufferV.handle, BufferAccessMode::read },
    { bufferR.handle, BufferAccessMode::write }
  };

  ShaderHandle shader = gpu->addShader("matrix_multiply", shaderCode, buffers,
    {{ SpecializationConstant::Type::uint_type, static_cast<uint32_t>(V.size()) }},
    { static_cast<uint32_t>(M.rows()), 1, 1 });

  gpu->queueShader(shader);
  gpu->flushQueue();

  gpu->retrieveBuffer(bufferR.handle, R.data());

  Vector expectedR = M * V;
  EXPECT_EQ(R, expectedR);
}

TEST_F(GpuTest, convolution) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);

  Array3 X({{
    { 4, 2, 5, 7, 4 },
    { 7, 3, 8, 2, 8 },
    { 9, 1, 6, 1, 2 },
    { 6, 3, 4, 8, 9 }
  }, {
    { 5, 4, 8, 1, 3 },
    { 3, 1, 7, 8, 9 },
    { 8, 2, 5, 2, 4 },
    { 3, 9, 8, 2, 5 }
  }});

  Kernel K({{
    { 5, 4 },
    { 1, 3 }
  }, {
    { 3, 7 },
    { 2, 4 }
  }});

  Array2 R(4, 3);

  GpuBuffer bufferX = gpu->allocateBuffer(X.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  GpuBuffer bufferK = gpu->allocateBuffer(K.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  GpuBuffer bufferR = gpu->allocateBuffer(R.size() * sizeof(netfloat_t), GpuBufferFlags::large);

  gpu->submitBufferData(bufferX.handle, X.data());
  gpu->submitBufferData(bufferK.handle, K.data());

  auto shaderCode = m_fileSystem->loadBinaryFile("test/shaders/convolution.spv");

  Size3 workgroupSize{
    static_cast<uint32_t>(R.W()),
    static_cast<uint32_t>(R.H()),
    1
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(K.W()) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(K.H()) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(K.D()) }
  };

  GpuBufferBindings buffers{
    { bufferX.handle, BufferAccessMode::read },
    { bufferK.handle, BufferAccessMode::read },
    { bufferR.handle, BufferAccessMode::write }
  };

  ShaderHandle shader = gpu->addShader("convolution", shaderCode, buffers, constants,
    workgroupSize);

  gpu->queueShader(shader);
  gpu->flushQueue();

  gpu->retrieveBuffer(bufferR.handle, R.data());

  Array2 expectedR(4, 3);
  computeConvolution(X, K, expectedR);

  for (size_t j = 0; j < expectedR.rows(); ++j) {
    for (size_t i = 0; i < expectedR.cols(); ++i) {
      EXPECT_NEAR(R.at(i, j), expectedR.at(i, j), FLOAT_TOLERANCE);
    }
  }
}

TEST_F(GpuTest, fullConvolution) {
  testing::NiceMock<MockLogger> logger;
  GpuPtr gpu = createGpu(logger);

  Array3 X({{
    { 4, 2, 5, 7, 4 },
    { 7, 3, 8, 2, 8 },
    { 9, 1, 6, 1, 2 },
    { 6, 3, 4, 8, 9 }
  }, {
    { 5, 4, 8, 1, 3 },
    { 3, 1, 7, 8, 9 },
    { 8, 2, 5, 2, 4 },
    { 3, 9, 8, 2, 5 }
  }});

  Kernel K({{
    { 5, 4 },
    { 1, 3 }
  }, {
    { 3, 7 },
    { 2, 4 }
  }});

  Array2 R(6, 5);

  GpuBuffer bufferX = gpu->allocateBuffer(X.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  GpuBuffer bufferK = gpu->allocateBuffer(K.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  GpuBuffer bufferR = gpu->allocateBuffer(R.size() * sizeof(netfloat_t), GpuBufferFlags::large);

  gpu->submitBufferData(bufferX.handle, X.data());
  gpu->submitBufferData(bufferK.handle, K.data());

  auto shaderCode = m_fileSystem->loadBinaryFile("test/shaders/full_convolution.spv");

  Size3 workgroupSize{
    static_cast<uint32_t>(R.W()),
    static_cast<uint32_t>(R.H()),
    1
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(K.W()) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(K.H()) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(K.D()) }
  };

  GpuBufferBindings buffers{
    { bufferX.handle, BufferAccessMode::read },
    { bufferK.handle, BufferAccessMode::read },
    { bufferR.handle, BufferAccessMode::write }
  };

  ShaderHandle shader = gpu->addShader("full_convolution", shaderCode, buffers, constants,
    workgroupSize);

  gpu->queueShader(shader);
  gpu->flushQueue();

  gpu->retrieveBuffer(bufferR.handle, R.data());

  Array2 expectedR(6, 5);
  computeFullConvolution(X, K, expectedR);

  for (size_t j = 0; j < expectedR.rows(); ++j) {
    for (size_t i = 0; i < expectedR.cols(); ++i) {
      EXPECT_NEAR(R.at(i, j), expectedR.at(i, j), FLOAT_TOLERANCE);
    }
  }
}
