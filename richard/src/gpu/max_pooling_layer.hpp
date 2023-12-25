#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace gpu {

class MaxPoolingLayer : public Layer {
  public:
    MaxPoolingLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputW, size_t inputH,
      size_t inputDepth);

    void allocateGpuResources(GpuBufferHandle statusBuffer, GpuBufferHandle inputBuffer,
      const Layer* nextLayer, GpuBufferHandle sampleYBuffer) override;
    size_t size() const override;
    GpuBufferHandle outputBuffer() const override;
    GpuBufferHandle weightsBuffer() const override;
    GpuBufferHandle deltaBuffer() const override;
    void retrieveBuffers() override;
    Triple outputSize() const override;
    void evalForward() override;
    void trainForward() override;
    void backprop() override;
    void updateParams() override;
    void writeToStream(std::ostream& stream) const override;

  private:
    // TODO
};

}
}
