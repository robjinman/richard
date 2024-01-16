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

  ShaderHandle shader = gpu->compileShader("simple_shader", shaderSource, { buffer.handle }, {},
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

  ShaderHandle shader = gpu->compileShader("structured_buffer", shaderSource,
    { statusBuffer.handle }, {}, { 16, 1, 1 });

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

  ShaderHandle shader = gpu->compileShader("matrix_multiply", shaderSource,
    { bufferM.handle, bufferV.handle, bufferR.handle },
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

    uint arrayIndex3d(uint W, uint H, uint x, uint y, uint z) {
      return z * W * H + y * W + x;
    }

    layout(constant_id = 0) const uint local_size_x = 1;
    layout(constant_id = 1) const uint local_size_y = 1;
    layout(constant_id = 2) const uint local_size_z = 1;

    layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

    layout(constant_id = 3) const uint KERNEL_W = 1;
    layout(constant_id = 4) const uint KERNEL_H = 1;
    layout(constant_id = 5) const uint KERNEL_D = 1;

    layout(std140, binding = 0) readonly buffer ImageSsbo {
      vec4 Image[];
    };

    FN_READ(Image)

    layout(std140, binding = 1) readonly buffer KernelSsbo {
      vec4 Kernel[];
    };

    FN_READ(Kernel)

    layout(std140, binding = 2) writeonly buffer ResultSsbo {
      vec4 Result[];
    };

    FN_WRITE(Result)

    void main() {
      const uint xIdx = gl_GlobalInvocationID.x;
      const uint yIdx = gl_GlobalInvocationID.y;

      const uint fmW = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
      const uint fmH = gl_WorkGroupSize.y * gl_NumWorkGroups.y;

      const uint imW = fmW + KERNEL_W - 1;
      const uint imH = fmH + KERNEL_H - 1;

      float sum = 0.0;
      for (uint k = 0; k < KERNEL_D; ++k) {
        for (uint j = 0; j < KERNEL_H; ++j) {
          for (uint i = 0; i < KERNEL_W; ++i) {
            const uint x = xIdx + i;
            const uint y = yIdx + j;
            const uint z = k;

            const float pixel = readImage(arrayIndex3d(imW, imH, x, y, z));
            const uint kernelIdx = arrayIndex3d(KERNEL_W, KERNEL_H, KERNEL_W - i - 1,
              KERNEL_H - j - 1, k);
            sum += pixel * readKernel(kernelIdx);
          }
        }
      }

      writeResult(yIdx * fmW + xIdx, sum);
    }
  )";

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

  ShaderHandle shader = gpu->compileShader("convolution", shaderSource,
    { bufferX.handle, bufferK.handle, bufferR.handle }, constants, workgroupSize);

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

    uint arrayIndex3d(uint W, uint H, uint x, uint y, uint z) {
      return z * W * H + y * W + x;
    }

    layout(constant_id = 0) const uint local_size_x = 1;
    layout(constant_id = 1) const uint local_size_y = 1;
    layout(constant_id = 2) const uint local_size_z = 1;

    layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

    layout(constant_id = 3) const uint KERNEL_W = 1;
    layout(constant_id = 4) const uint KERNEL_H = 1;
    layout(constant_id = 5) const uint KERNEL_D = 1;

    layout(std140, binding = 0) readonly buffer ImageSsbo {
      vec4 Image[];
    };

    FN_READ(Image)

    layout(std140, binding = 1) readonly buffer KernelSsbo {
      vec4 Kernel[];
    };

    FN_READ(Kernel)

    layout(std140, binding = 2) writeonly buffer ResultSsbo {
      vec4 Result[];
    };

    FN_WRITE(Result)

    void main() {
      const int xIdx = int(gl_GlobalInvocationID.x);
      const int yIdx = int(gl_GlobalInvocationID.y);

      const int fmW = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);
      const int fmH = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);

      const int kW = int(KERNEL_W);
      const int kH = int(KERNEL_H);

      const int imW = fmW - kW + 1;
      const int imH = fmH - kH + 1;

      const int xMin = -kW + 1;
      const int yMin = -kH + 1;

      float sum = 0.0;
      for (int k = 0; k < KERNEL_D; ++k) {
        for (int j = 0; j < kH; ++j) {
          for (int i = 0; i < kW; ++i) {
            const int x = xMin + xIdx + i;
            const int y = yMin + yIdx + j;

            if (x < 0 || x + 1 > imW) {
              continue;
            }

            if (y < 0 || y + 1 > imH) {
              continue;
            }

            const float pixel = readImage(arrayIndex3d(imW, imH, x, y, k));
            const uint kernelIdx = arrayIndex3d(KERNEL_W, KERNEL_H, kW - i - 1, kH - j - 1, k);
            sum += pixel * readKernel(kernelIdx);
          }
        }
      }

      writeResult(yIdx * fmW + xIdx, sum);
    }
  )";

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

  ShaderHandle shader = gpu->compileShader("full_convolution", shaderSource,
    { bufferX.handle, bufferK.handle, bufferR.handle }, constants, workgroupSize);

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
