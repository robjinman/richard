#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace gpu {

class OutputLayer : public Layer {
  public:
    OutputLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize);
    OutputLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream, size_t inputSize);

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
    const Vector& activations() const;

    // Exposed for testing
    //
    void test_setWeights(const DataArray& W);
    void test_setBiases(const DataArray& B);
    GpuBufferHandle test_deltaWBuffer() const;
    GpuBufferHandle test_deltaBBuffer() const;
    const Matrix& test_W() const;
    const Vector& test_B() const;

  private:
    void initialize(const nlohmann::json& obj, size_t inputSize);

    Gpu& m_gpu;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    size_t m_inputSize;
    size_t m_size;
    Vector m_B;
    Matrix m_W;
    mutable Vector m_A;
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
