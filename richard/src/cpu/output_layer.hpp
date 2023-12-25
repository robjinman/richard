#pragma once

#include "cpu/layer.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace cpu {

class OutputLayer : public Layer {
  public:
    OutputLayer(const nlohmann::json& obj, size_t inputSize);
    OutputLayer(const nlohmann::json& obj, std::istream& stream, size_t inputSize);

    LayerType type() const override { return LayerType::OUTPUT; }
    Triple outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& delta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDelta(const DataArray&, const Layer&) override;
    void updateDelta(const DataArray& inputs, const DataArray& outputs);
    void updateParams(size_t epoch) override;
    void writeToStream(std::ostream& stream) const override;
    const Matrix& W() const override;

    // Exposed for testing
    //
    void setWeights(const Matrix& weights);
    void setBiases(const Vector& biases);
    void setWeights(const DataArray& W);
    void setBiases(const DataArray& B);
    void setActivationFn(ActivationFn f, ActivationFn fPrime);

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    Vector m_deltaB;
    Matrix m_deltaW;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    ActivationFn m_activationFn;
    ActivationFn m_activationFnPrime;
};

}
}
