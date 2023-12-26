#include "mock_logger.hpp"
#include <types.hpp>
#include <gpu/gpu.hpp>
#include <gtest/gtest.h>

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
