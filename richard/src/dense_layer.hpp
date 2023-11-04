#pragma once

#include "layer.hpp"

class DenseLayer : public Layer {
  public:
    DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize);
    DenseLayer(const nlohmann::json& obj, size_t inputSize);

    LayerType type() const override { return LayerType::DENSE; }
    Triple outputSize() const override;
    const DataArray& activations() const override;
    const DataArray& delta() const override;
    void trainForward(const DataArray& inputs) override;
    DataArray evalForward(const DataArray& inputs) const override;
    void updateDelta(const DataArray& inputs, const Layer& nextLayer, size_t epoch) override;
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
    double m_dropoutRate;
    ActivationFn m_activationFn;
    ActivationFn m_activationFnPrime;
};

