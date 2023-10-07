#pragma once

#include "layer.hpp"

class ConvolutionalLayer : public Layer {
  public:
    ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH);
    ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW, size_t inputH);

    LayerType type() const override { return LayerType::CONVOLUTIONAL; }
    std::array<size_t, 3> outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer, size_t epoch) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    const Matrix& W() const override;

  private:
    struct SliceParams {
      SliceParams()
        : W(1, 1)
        , b(0.0) {}

      Matrix W;
      double b;
    };

    std::vector<SliceParams> m_slices;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    size_t m_inputW;
    size_t m_inputH;
    double m_learnRate;
    double m_learnRateDecay;

  void forwardPass(const Vector& inputs, Vector& Z) const;
  size_t sliceSize() const;
};
