#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace gpu {

class DenseLayer : public Layer {
  public:
    DenseLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize, bool isFirstLayer);
    DenseLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream, size_t inputSize,
      bool isFirstLayer);

    void allocateGpuBuffers() override;
    void createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
      const Layer* nextLayer, GpuBufferHandle sampleYBuffer) override;
    size_t size() const override;
    GpuBufferHandle outputBuffer() const override;
    GpuBufferHandle weightsBuffer() const override;
    GpuBufferHandle deltaBuffer() const override;
    GpuBufferHandle inputDeltaBuffer() const override;
    void retrieveBuffers() override;
    Size3 outputSize() const override;
    void evalForward() override;
    void trainForward() override;
    void backprop() override;
    void updateParams() override;
    void writeToStream(std::ostream& stream) const override;

    // Exposed for testing
    //
    void test_setWeights(const DataArray& W);
    void test_setBiases(const DataArray& B);
    GpuBufferHandle test_activationsBuffer() const;
    GpuBufferHandle test_deltaWBuffer() const;
    GpuBufferHandle test_deltaBBuffer() const;
    const Matrix& test_W() const;
    const Vector& test_B() const;

  private:
    void initialize(const nlohmann::json& obj, size_t inputSize, bool isFirstLayer);

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
