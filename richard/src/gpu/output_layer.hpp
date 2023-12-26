#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace gpu {

class OutputLayer : public Layer {
  public:
    OutputLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream, size_t inputSize,
      size_t miniBatchSize);
    OutputLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize, size_t miniBatchSize);

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
    void setWeights(const Matrix& weights);
    void setBiases(const Vector& biases);
    GpuBufferHandle activationsBuffer() const;

  private:
    Gpu& m_gpu;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    size_t m_miniBatchSize;
    size_t m_inputSize;
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
