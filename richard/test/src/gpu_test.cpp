#include "mock_logger.hpp"
#include <types.hpp>
#include <math.hpp>
#include <gpu/gpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>

using namespace richard;
using namespace richard::gpu;

const double FLOAT_TOLERANCE = 0.0001;

class GpuTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
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
    data[i] = i;
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
    data[i] = i;
  }

  GpuBuffer buffer = gpu->allocateBuffer(data.size() * sizeof(netfloat_t), GpuBufferFlags::large);
  gpu->submitBufferData(buffer.handle, data.data());

  std::string shaderSource = R"(
    #version 430

    #define FN_READ(BUF) \
      float read##BUF(uint pos) { \
        return BUF[pos / 4][pos % 4]; \
      }

    #define FN_WRITE(BUF) \
      void write##BUF(uint pos, float val) { \
        BUF[pos / 4][pos % 4] = val; \
      }

    layout(constant_id = 0) const uint local_size_x = 1;
    layout(constant_id = 1) const uint local_size_y = 1;
    layout(constant_id = 2) const uint local_size_z = 1;

    layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

    layout(std140, binding = 0) buffer Ssbo {
      vec4 X[];
    };

    FN_READ(X)
    FN_WRITE(X)

    void main() {
      const uint index = gl_GlobalInvocationID.x;

      writeX(index, readX(index) * 2.0);
    }
  )";

  ShaderHandle shader = gpu->compileShader(shaderSource, { buffer.handle }, {},
    { bufferSize, 1, 1 });

  gpu->queueShader(shader);
  gpu->flushQueue();

  std::array<netfloat_t, bufferSize> expected{};
  std::transform(data.begin(), data.end(), expected.begin(), [](netfloat_t x) { return x * 2.0; });

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

  std::string shaderSource = R"(
    #version 430

    struct StatusBuffer {
      uint epoch;
      float cost;
      uint sampleIndex;
    };

    layout(constant_id = 0) const uint local_size_x = 1;
    layout(constant_id = 1) const uint local_size_y = 1;
    layout(constant_id = 2) const uint local_size_z = 1;

    layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

    layout(std140, binding = 0) buffer StatusSsbo {
      StatusBuffer Status;
    };

    void main() {
      const uint index = gl_GlobalInvocationID.x;

      if (index == 0) {
        Status.cost = Status.cost + 123.45;
      }
    }
  )";

  ShaderHandle shader = gpu->compileShader(shaderSource, { statusBuffer.handle }, {}, { 16, 1, 1 });

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

  std::string shaderSource = R"(
    #version 430

    #define FN_READ(BUF) \
      float read##BUF(uint pos) { \
        return BUF[pos / 4][pos % 4]; \
      }

    #define FN_WRITE(BUF) \
      void write##BUF(uint pos, float val) { \
        BUF[pos / 4][pos % 4] = val; \
      }

    layout(constant_id = 0) const uint local_size_x = 1;
    layout(constant_id = 1) const uint local_size_y = 1;
    layout(constant_id = 2) const uint local_size_z = 1;

    layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

    layout(constant_id = 3) const uint VECTOR_SIZE = 1;

    layout(std140, binding = 0) readonly buffer MSsbo {
      vec4 M[];
    };

    FN_READ(M)

    layout(std140, binding = 1) readonly buffer VSsbo {
      vec4 V[];
    };

    FN_READ(V)

    layout(std140, binding = 2) writeonly buffer RSsbo {
      vec4 R[];
    };

    FN_WRITE(R)

    void main() {
      const uint index = gl_GlobalInvocationID.x;

      float weightedSum = 0.0;
      for (uint i = 0; i < VECTOR_SIZE; ++i) {
        float m = readM(index * VECTOR_SIZE + i);
        float v = readV(i);
        weightedSum += m * v;
      }
      writeR(index, weightedSum);
    }
  )";

  ShaderHandle shader = gpu->compileShader(shaderSource,
    { bufferM.handle, bufferV.handle, bufferR.handle },
    {{ SpecializationConstant::Type::uint_type, static_cast<uint32_t>(V.size()) }},
    { static_cast<uint32_t>(M.rows()), 1, 1 });

  gpu->queueShader(shader);
  gpu->flushQueue();

  gpu->retrieveBuffer(bufferR.handle, R.data());

  Vector expectedR = M * V;
  EXPECT_EQ(R, expectedR);
}
