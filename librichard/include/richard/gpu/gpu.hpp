#pragma once

#include "richard/types.hpp"
#include "richard/config.hpp"
#include <string>
#include <memory>
#include <vector>
#include <array>

namespace richard {

class Logger;

namespace gpu {

using ShaderHandle = uint32_t;
using GpuBufferHandle = uint32_t;

using ShaderCode = std::vector<uint8_t>;

enum class BufferAccessMode {
  read,
  write
};

struct BufferBinding {
  GpuBufferHandle buffer;
  BufferAccessMode mode;
};

using GpuBufferBindings = std::vector<BufferBinding>;

struct SpecializationConstant {
  enum class Type {
    uint_type,
    float_type,
    bool_type
  };

  Type type;
  std::variant<uint32_t, float, bool> value;
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
    virtual ShaderHandle addShader(const std::string& name, const ShaderCode& shaderCode,
      const GpuBufferBindings& bufferBindings, const SpecializationConstants& constants,
      uint32_t pushConstantsSize, const Size3& workSize) = 0;
    virtual void submitBufferData(GpuBufferHandle buffer, const void* data) = 0;
    virtual void queueShader(ShaderHandle shaderHandle, const void* pushConstants = nullptr) = 0;
    virtual void retrieveBuffer(GpuBufferHandle buffer, void* data) = 0;
    virtual void flushQueue() = 0;

    virtual ~Gpu() = default;
};

using GpuPtr = std::unique_ptr<Gpu>;

GpuPtr createGpu(Logger& logger, const Config& config = Config{});

}
}
