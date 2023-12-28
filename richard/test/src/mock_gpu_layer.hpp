#pragma once

#include <gpu/layer.hpp>
#include <gmock/gmock.h>

using namespace richard;
using namespace richard::gpu;

class MockGpuLayer : public gpu::Layer {
  public:
    MOCK_METHOD(void, allocateGpuBuffers, (), (override));
    MOCK_METHOD(void, createGpuShaders, (GpuBufferHandle statusBuffer,
      GpuBufferHandle inputBuffer, const Layer* nextLayer, GpuBufferHandle sampleYBuffer),
      (override));
    MOCK_METHOD(size_t, size, (), (const, override));
    MOCK_METHOD(GpuBufferHandle, outputBuffer, (), (const, override));
    MOCK_METHOD(GpuBufferHandle, weightsBuffer, (), (const, override));
    MOCK_METHOD(GpuBufferHandle, deltaBuffer, (), (const, override));
    MOCK_METHOD(void, retrieveBuffers, (), (override));
    MOCK_METHOD(Triple, outputSize, (), (const, override));
    MOCK_METHOD(void, evalForward, (), (override));
    MOCK_METHOD(void, trainForward, (), (override));
    MOCK_METHOD(void, backprop, (), (override));
    MOCK_METHOD(void, updateParams, (), (override));
    MOCK_METHOD(void, writeToStream, (std::ostream& stream), (const, override));
};

