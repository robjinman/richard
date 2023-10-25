#pragma once

#include <vector>
#include "layer.hpp"

class ConvolutionalLayer : public Layer {
  public:
    ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH, size_t inputDepth);
    ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW, size_t inputH,
      size_t inputDepth);

    LayerType type() const override { return LayerType::CONVOLUTIONAL; }
    std::array<size_t, 3> outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& delta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDelta(const DataArray& inputs, const Layer& nextLayer, size_t epoch) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    // Don't use. Use kernel() instead.
    const Matrix& W() const override;

    std::array<size_t, 2> kernelSize() const;
    size_t depth() const;

    // Exposed for testing
    //

    struct Filter {
      Filter()
        : K(1, 1, 1)
        , b(0.0) {}

      Kernel K;
      double b;
    };

    void forwardPass(const Array3& inputs, Array3& Z) const;
    void setFilters(const std::vector<ConvolutionalLayer::Filter>& filters);
    const std::vector<Filter>& filters() const;
    void setWeights(const std::vector<DataArray>&) override;
    void setBiases(const DataArray&) override;

  private:
    std::vector<Filter> m_filters;
    Array3 m_Z;
    Array3 m_A;
    Array3 m_delta;
    size_t m_inputW;
    size_t m_inputH;
    size_t m_inputDepth;
    double m_learnRate;
    double m_learnRateDecay;

    size_t numOutputs() const;
};

