#pragma once

#include "layer.hpp"

class OutputLayer : public Layer {
  public:
    OutputLayer(const nlohmann::json& obj, size_t inputSize);
    OutputLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize);

    LayerType type() const override { return LayerType::OUTPUT; }
    Triple outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& delta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDelta(const DataArray&, const Layer&, size_t) override { assert(false); }
    void updateDelta(const DataArray& inputs, const DataArray& y, size_t epoch);
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    const Matrix& W() const override;

    // Exposed for testing
    //
    void setWeights(const Matrix& weights);
    void setBiases(const Vector& biases);
    void setWeights(const std::vector<DataArray>& W) override;
    void setBiases(const DataArray& B) override;
    void setActivationFn(ActivationFn f, ActivationFn fPrime);

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    double m_learnRate;
    double m_learnRateDecay;
    ActivationFn m_activationFn;
    ActivationFn m_activationFnPrime;
};
