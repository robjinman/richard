#pragma once

#include "cpu/layer.hpp"
#include "cpu/convolutional_layer.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace cpu {

class MaxPoolingLayer : public Layer {
  public:
    MaxPoolingLayer(const nlohmann::json& obj, const Size3& inputShape);

    LayerType type() const override { return LayerType::MAX_POOLING; }
    Size3 outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& delta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDelta(const DataArray& inputs, const Layer& nextLayer) override;
    void updateParams(size_t) override {}
    void writeToStream(std::ostream&) const override {}
    const Matrix& W() const override;

    // Exposed for testing
    //
    void test_padDelta(const Array3& delta, const Array3& mask, Array3& paddedDelta) const;
    const Array3& test_mask() const;
    void test_backpropFromConvLayer(const std::vector<ConvolutionalLayer::Filter>& filters,
      const DataArray& convDelta, Array3& delta);

  private:
    void padDelta(const Array3& delta, const Array3& mask, Array3& paddedDelta) const;
    void backpropFromDenseLayer(const Layer& nextLayer, Array3& delta);
    void backpropFromConvLayer(const std::vector<ConvolutionalLayer::Filter>& filters,
      const DataArray& convDelta, Array3& delta);

    Array3 m_Z;
    Array3 m_delta;
    Array3 m_paddedDelta;
    size_t m_regionW;
    size_t m_regionH;
    size_t m_inputW;
    size_t m_inputH;
    size_t m_inputDepth;
    Array3 m_mask;
};

}
}
