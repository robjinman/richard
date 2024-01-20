#pragma once

#include "math.hpp"
#include "gpu/layer.hpp"
#include "gpu/gpu.hpp"

namespace richard {

class FileSystem;
class PlatformPaths;
class Config;

namespace gpu {

class MaxPoolingLayer : public Layer {
  public:
    MaxPoolingLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
      const Config& config, const Size3& inputShape);

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
    GpuBufferHandle test_maskBuffer() const;

  private:
    void createEvalForwardShader(GpuBufferHandle inputBuffer);
    void createTrainForwardShader(GpuBufferHandle inputBuffer);
    void createBackpropShader(const Layer* nextLayer);

    Gpu& m_gpu;
    FileSystem& m_fileSystem;
    const PlatformPaths& m_platformPaths;
    size_t m_regionW;
    size_t m_regionH;
    size_t m_inputW;
    size_t m_inputH;
    size_t m_inputDepth;
    GpuBuffer m_bufferZ;
    GpuBuffer m_bufferMask;
    GpuBuffer m_bufferInputDelta;
    ShaderHandle m_evalForwardShader;
    ShaderHandle m_trainForwardShader;
    ShaderHandle m_backpropShader;
};

}
}
