#pragma once

#include "gpu/gpu.hpp"
#include "types.hpp"
#include <fstream>
#include <memory>

namespace richard {
namespace gpu {

class Layer {
  public:
    virtual void allocateGpuBuffers() = 0;
    virtual void createGpuShaders(GpuBufferHandle statusBuffer, GpuBufferHandle inputBuffer,
      const Layer* nextLayer, GpuBufferHandle sampleYBuffer) = 0;
    virtual size_t size() const = 0;
    virtual GpuBufferHandle outputBuffer() const = 0;
    virtual GpuBufferHandle weightsBuffer() const = 0;
    virtual GpuBufferHandle deltaBuffer() const = 0;
    virtual void retrieveBuffers() = 0;
    virtual Triple outputSize() const = 0;
    virtual void evalForward() = 0;
    virtual void trainForward() = 0;
    virtual void backprop() = 0;
    virtual void updateParams() = 0;
    virtual void writeToStream(std::ostream& stream) const = 0;

    virtual ~Layer() = default;
};

using LayerPtr = std::unique_ptr<Layer>;

}
}
