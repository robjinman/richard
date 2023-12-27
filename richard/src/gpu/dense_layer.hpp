#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace gpu {

class DenseLayer : public Layer {
  public:
    DenseLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream, size_t inputSize,
      bool isFirstLayer);
    DenseLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize, bool isFirstLayer);

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

    // Exposed for testing
    //
    void setWeights(const DataArray& W);
    void setBiases(const DataArray& B);
    GpuBufferHandle activationsBuffer() const;

  private:
    Gpu& m_gpu;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    netfloat_t m_dropoutRate;
    size_t m_inputSize;
    bool m_isFirstLayer;
    size_t m_size;
    Vector m_B;
    Matrix m_W;
    GpuBuffer m_bufferB;
    GpuBuffer m_bufferW;
    GpuBuffer m_bufferZ;
    GpuBuffer m_bufferA;
    GpuBuffer m_bufferD;
    GpuBuffer m_bufferDeltaB;
    GpuBuffer m_bufferDeltaW;
    ShaderHandle m_evalForwardShader;
    ShaderHandle m_trainForwardShader;
    ShaderHandle m_backpropShader;
    ShaderHandle m_updateParamsShader;
};

}
}
