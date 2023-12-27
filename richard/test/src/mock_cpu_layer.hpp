#pragma once

#include <cpu/layer.hpp>
#include <gmock/gmock.h>

using namespace richard;
using namespace richard::cpu;

class MockCpuLayer : public cpu::Layer {
  public:
    MOCK_METHOD(LayerType, type, (), (const, override));
    MOCK_METHOD(Triple, outputSize, (), (const, override));
    MOCK_METHOD(const DataArray&, activations, (), (const, override));
    MOCK_METHOD(const DataArray&, delta, (), (const, override));
    MOCK_METHOD(void, trainForward, (const DataArray& inputs), (override));
    MOCK_METHOD(DataArray, evalForward, (const DataArray& inputs), (const, override));
    MOCK_METHOD(void, updateDelta, (const DataArray& inputs, const Layer& nextLayer), (override));
    MOCK_METHOD(void, updateParams, (size_t epoch), (override));
    MOCK_METHOD(void, writeToStream, (std::ostream& fout), (const, override));
    MOCK_METHOD(const Matrix&, W, (), (const, override));
};

