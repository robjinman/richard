#pragma once

#include <cpu/layer.hpp>
#include <gmock/gmock.h>

using namespace richard;
using namespace richard::cpu;

class MockCpuLayer : public cpu::Layer {
  public:
    MOCK_METHOD(Size3, outputSize, (), (const, override));
    MOCK_METHOD(const DataArray&, activations, (), (const, override));
    MOCK_METHOD(const DataArray&, inputDelta, (), (const, override));
    MOCK_METHOD(void, trainForward, (const DataArray& inputs), (override));
    MOCK_METHOD(DataArray, evalForward, (const DataArray& inputs), (const, override));
    MOCK_METHOD(void, updateDeltas, (const DataArray& inputs, const DataArray& outputDelta),
      (override));
    MOCK_METHOD(void, updateParams, (size_t epoch), (override));
    MOCK_METHOD(void, writeToStream, (std::ostream& fout), (const, override));
};

