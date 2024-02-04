#pragma once

#include "cpu/layer.hpp"
#include <vector>

namespace richard {

class Config;

namespace cpu {

class ConvolutionalLayer : public Layer {
  public:
    struct Filter {
      Kernel K;
      netfloat_t b;
    };

    ConvolutionalLayer(const Config& config, const Size3& inputShape);
    ConvolutionalLayer(const Config& config, std::istream& stream, const Size3& inputShape);

    Size3 outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& inputDelta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDeltas(const DataArray& inputs, const DataArray& outputDelta) override;
    void updateParams(size_t epoch) override;
    void writeToStream(std::ostream& stream) const override;

    // Exposed for testing
    //
    void test_setFilters(const std::vector<Filter>& filters);
    const std::vector<Filter> test_filters() const;
    const std::vector<Filter> test_filterDeltas() const;

  private:
    void initialize(const Config& config, const Size3& inputShape);
    size_t numOutputs() const;
    void forwardPass(const Array3& inputs, Array3& Z) const;

    std::vector<Filter> m_filters;
    Array3 m_Z;
    Array3 m_A;
    Array3 m_inputDelta;
    std::vector<Filter> m_paramDeltas;
    size_t m_inputW;
    size_t m_inputH;
    size_t m_inputDepth;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    netfloat_t m_dropoutRate;
};

}
}
