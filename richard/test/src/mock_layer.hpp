#pragma once

#include <gmock/gmock.h>
#include <layer.hpp>

using Array3 = std::array<size_t, 3>;

class MockLayer : public Layer {
  public:
    MOCK_METHOD(LayerType, type, (), (const, override));
    MOCK_METHOD(Array3, outputSize, (), (const, override));
    MOCK_METHOD(const Vector&, activations, (), (const, override));
    MOCK_METHOD(const Vector&, delta, (), (const, override));
    MOCK_METHOD(void, trainForward, (const Vector& inputs), (override));
    MOCK_METHOD(Vector, evalForward, (const Vector& inputs), (const, override));
    MOCK_METHOD(void, updateDelta, (const Vector& layerInputs, const Layer& nextLayer,
      size_t epoch), (override));
    MOCK_METHOD(nlohmann::json, getConfig, (), (const, override));
    MOCK_METHOD(void, writeToStream, (std::ostream& fout), (const, override));
    MOCK_METHOD(const Matrix&, W, (), (const, override));
};
