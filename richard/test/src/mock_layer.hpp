#pragma once

#include <gmock/gmock.h>
#include <layer.hpp>

using StdArray3 = std::array<size_t, 3>;

class MockLayer : public Layer {
  public:
    MOCK_METHOD(LayerType, type, (), (const, override));
    MOCK_METHOD(StdArray3, outputSize, (), (const, override));
    MOCK_METHOD(const DataArray&, activations, (), (const, override));
    MOCK_METHOD(const DataArray&, delta, (), (const, override));
    MOCK_METHOD(void, trainForward, (const DataArray& inputs), (override));
    MOCK_METHOD(DataArray, evalForward, (const DataArray& inputs), (const, override));
    MOCK_METHOD(void, updateDelta, (const DataArray& inputs, const Layer& nextLayer,
      size_t epoch), (override));
    MOCK_METHOD(nlohmann::json, getConfig, (), (const, override));
    MOCK_METHOD(void, writeToStream, (std::ostream& fout), (const, override));
    MOCK_METHOD(const Matrix&, W, (), (const, override));
    MOCK_METHOD(void, setWeights, (const std::vector<DataArray>&), (override));
    MOCK_METHOD(void, setBiases, (const DataArray&), (override));
};

/*
    virtual LayerType type() const = 0;
    virtual std::array<size_t, 3> outputSize() const = 0;
    virtual const DataArray& activations() const = 0;
    virtual const DataArray& delta() const = 0;
    virtual void trainForward(const DataArray& inputs) = 0;
    virtual DataArray evalForward(const DataArray& inputs) const = 0;
    virtual void updateDelta(const DataArray& inputs, const Layer& nextLayer, size_t epoch) = 0;
    virtual nlohmann::json getConfig() const = 0;
    virtual void writeToStream(std::ostream& fout) const = 0;
    virtual const Matrix& W() const = 0;
*/
