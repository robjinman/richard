#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"

namespace richard {

class FileSystem;
class PlatformPaths;
class Config;

namespace gpu {

class ConvolutionalLayer : public Layer {
  public:
    ConvolutionalLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
      const Config& config, const Size3& inputShape, bool isFirstLayer);
    ConvolutionalLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
      const Config& config, std::istream& stream, const Size3& inputShape, bool isFirstLayer);

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
    void test_setKernels(const DataArray& kernelData);
    void test_setBiases(const DataArray& biasData);
    GpuBufferHandle test_deltaKBuffer() const;
    GpuBufferHandle test_deltaBBuffer() const;
    const DataArray& test_kernels() const;
    const Vector& test_biases() const;

  private:
    void initialize(const Config& config, const Size3& inputShape, bool isFirstLayer);
    void createEvalForwardShader(GpuBufferHandle inputBuffer);
    void createTrainForwardShader(GpuBufferHandle statusBuffer, GpuBufferHandle inputBuffer);
    void createBackpropDeltaShader(const Layer* nextLayer);
    void createBackpropInputDeltaShader();
    void createBackpropParamDeltasShader(GpuBufferHandle statusBuffer, GpuBufferHandle inputBuffer);
    void createUpdateParamsShader(GpuBufferHandle statusBuffer);

    Gpu& m_gpu;
    FileSystem& m_fileSystem;
    const PlatformPaths& m_platformPaths;
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
    GpuBuffer m_bufferInputDelta;
    GpuBuffer m_bufferDeltaK;
    GpuBuffer m_bufferDeltaB;
    ShaderHandle m_evalForwardShader;
    ShaderHandle m_trainForwardShader;
    ShaderHandle m_backpropDeltaShader;
    ShaderHandle m_backpropInputDeltaShader;
    ShaderHandle m_backpropParamDeltasShader;
    ShaderHandle m_updateParamsShader;
};

}
}
