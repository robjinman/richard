#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace gpu {

class ConvolutionalLayer : public Layer {
  public:
    ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj, const Size3& inputShape,
      bool isFirstLayer);
    ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream,
      const Size3& inputShape, bool isFirstLayer);

    void allocateGpuBuffers() override;
    void createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
      const Layer* nextLayer, GpuBufferHandle sampleYBuffer) override;
    size_t size() const override;
    GpuBufferHandle outputBuffer() const override;
    GpuBufferHandle weightsBuffer() const override;
    GpuBufferHandle deltaBuffer() const override;
    void retrieveBuffers() override;
    Size3 outputSize() const override;
    void evalForward() override;
    void trainForward() override;
    void backprop() override;
    void updateParams() override;
    void writeToStream(std::ostream& stream) const override;

  private:
    Gpu& m_gpu;
    size_t m_inputW;
    size_t m_inputH;
    size_t m_inputDepth;
    std::array<size_t, 2> m_kernelSize;
    size_t m_depth;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    netfloat_t m_dropoutRate;
    bool m_isFirstLayer;
    Vector m_kernelData;
    Vector m_biasData;
    GpuBuffer m_bufferK;
    GpuBuffer m_bufferB;
    GpuBuffer m_bufferZ;
    GpuBuffer m_bufferA;
    GpuBuffer m_bufferD;
    GpuBuffer m_bufferDeltaK;
    GpuBuffer m_bufferDeltaB;
    ShaderHandle m_evalForwardShader;
    ShaderHandle m_trainForwardShader;
    ShaderHandle m_backpropShader;
    ShaderHandle m_updateParamsShader;
};

}
}
