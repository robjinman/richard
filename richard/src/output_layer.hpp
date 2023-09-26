#pragma once

#include "layer.hpp"

class OutputLayer : public Layer {
  public:
    OutputLayer(size_t numNeurons, size_t inputSize, double learnRate);
    OutputLayer(std::istream& fin, size_t numNeurons, size_t inputSize, double learnRate);

    LayerType type() const override { return LayerType::OUTPUT; }
    std::array<size_t, 2> outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector&, const Layer&) override { assert(false); }
    void updateDelta(const Vector& layerInputs, const Vector& y);
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    const Matrix& W() const override;

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    double m_learnRate;
};
