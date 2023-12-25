#pragma once

#include <string>
#include <memory>
#include <vector>
#include <array>
#include <variant>

namespace richard {
namespace gpu {

using ShaderHandle = uint32_t;
using GpuBufferHandle = uint32_t;
using Size3 = const std::array<uint32_t, 3>;
using GpuBufferBindings = std::vector<GpuBufferHandle>;

struct SpecializationConstant {
  enum class Type {
    uint_type,
    float_type
  };

  Type type;
  std::variant<uint32_t, float> value;
};

using SpecializationConstants = std::vector<SpecializationConstant>;

enum class GpuBufferFlags {
  frequentHostAccess  = 1 << 0,
  hostReadAccess      = 1 << 1,
  hostWriteAccess     = 1 << 2,
  large               = 1 << 3,
  shaderReadonly      = 1 << 4
};

constexpr GpuBufferFlags operator|(GpuBufferFlags a, GpuBufferFlags b) {
  return static_cast<GpuBufferFlags>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr GpuBufferFlags operator&(GpuBufferFlags a, GpuBufferFlags b) {
  return static_cast<GpuBufferFlags>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr bool operator!(GpuBufferFlags flag) {
  return flag == static_cast<GpuBufferFlags>(0);
}

struct GpuBuffer {
  GpuBufferHandle handle = 0;
  size_t size = 0;
  uint8_t* data = nullptr;
};

class Gpu {
  public:
    virtual GpuBuffer allocateBuffer(size_t size, GpuBufferFlags flags) = 0;
    virtual ShaderHandle compileShader(const std::string& sourcePath,
      const GpuBufferBindings& bufferBindings, const SpecializationConstants& constants,
      const Size3& workgroupSize) = 0;
    virtual void submitBufferData(GpuBufferHandle buffer, const void* data) = 0;
    virtual void queueShader(ShaderHandle shaderHandle) = 0;
    virtual void retrieveBuffer(GpuBufferHandle buffer, void* data) = 0;
    virtual void flushQueue() = 0;

    virtual ~Gpu() = default;
};

using GpuPtr = std::unique_ptr<Gpu>;

GpuPtr createGpu();

}
}
