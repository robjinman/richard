#pragma once

#include <string>
#include <memory>
#include <vector>

using ShaderHandle = size_t;

class Gpu {
  public:
    virtual ShaderHandle compileShader(const std::string& source) = 0;
    virtual void submitBuffer(const void* buffer, size_t bufferSize) = 0;
    virtual void executeShader(size_t shaderIndex, size_t numWorkgroups) = 0;
    virtual void retrieveBuffer(void* data) = 0;

    virtual ~Gpu() {}
};

using GpuPtr = std::unique_ptr<Gpu>;

GpuPtr createGpu();
