#include "mock_logger.hpp"
#include <types.hpp>
#include <gpu/gpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>

using namespace richard;
using namespace richard::gpu;

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

  ASSERT_EQ(data, expected);
}