#pragma once

#include "richard/cpu/layer.hpp"
#include "richard/cpu/convolutional_layer.hpp"

namespace richard {

class Config;

namespace cpu {

class MaxPoolingLayer : public Layer {
  public:
    MaxPoolingLayer(const Config& config, const Size3& inputShape);

    Size3 outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& inputDelta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDeltas(const DataArray& inputs, const DataArray& outputDelta) override;
    void updateParams(size_t) override {}
    void writeToStream(std::ostream&) const override {}

    // Exposed for testing
    //
    void test_setMask(const Array3& mask);
    const Array3& test_mask() const;

  private:
    void padDelta(const Array3& delta, const Array3& mask, Array3& paddedDelta) const;
    void backpropFromDenseLayer(const Layer& nextLayer, Array3& delta);
    void backpropFromConvLayer(const std::vector<ConvolutionalLayer::Filter>& filters,
      const DataArray& convDelta, Array3& delta);

    Array3 m_Z;
    Array3 m_inputDelta;
    size_t m_regionW;
    size_t m_regionH;
    size_t m_inputW;
    size_t m_inputH;
    size_t m_inputDepth;
    Array3 m_mask;
};

}
}
