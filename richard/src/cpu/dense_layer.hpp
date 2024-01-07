#pragma once

#include "cpu/layer.hpp"
#include <nlohmann/json.hpp>

namespace richard {
namespace cpu {

class DenseLayer : public Layer {
  public:
    DenseLayer(const nlohmann::json& obj, size_t inputSize);
    DenseLayer(const nlohmann::json& obj, std::istream& stream, size_t inputSize);

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
    void test_setWeights(const DataArray& W);
    void test_setBiases(const DataArray& B);
    void test_setActivationFn(ActivationFn f, ActivationFn fPrime);
    const Matrix& test_deltaW() const;
    const Vector& test_deltaB() const;
    const Matrix& test_W() const;
    const Vector& test_B() const;

  private:
    void initialize(const nlohmann::json& obj, size_t inputSize);

    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_inputDelta;
    Vector m_deltaB;
    Matrix m_deltaW;
    netfloat_t m_learnRate;
    netfloat_t m_learnRateDecay;
    netfloat_t m_dropoutRate;
    ActivationFn m_activationFn;
    ActivationFn m_activationFnPrime;
};

}
}
