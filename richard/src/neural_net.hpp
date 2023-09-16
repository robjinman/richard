#pragma once

#include "math.hpp"

class LabelledDataSet;

class NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    explicit NeuralNet(const nlohmann::json& config);
    explicit NeuralNet(std::istream& s);

    CostFn costFn() const;
    size_t inputSize() const;
    void writeToStream(std::ostream& s) const;
    void train(LabelledDataSet& data);
    Vector evaluate(const Vector& inputs) const;

    // For unit tests
    void setWeights(const std::vector<Matrix>& W);
    void setBiases(const std::vector<Vector>& B);

    static const nlohmann::json& defaultConfig();

  private:
    double feedForward(const Vector& x, const Vector& y, double dropoutRate);
    void updateLayer(size_t layerIdx, const Vector& delta, const Vector& x, double learnRate);

    struct Layer {
      Matrix weights;
      Vector biases;
      Vector Z;
      Vector A;

      Layer(Layer&& mv);
      Layer(Matrix&& weights, Vector&& biases);
      Layer(const Layer& cpy);
      Layer(const Matrix& weights, const Vector& biases);
    };

    struct Params {
      Params();
      explicit Params(const nlohmann::json& obj);

      std::vector<size_t> layers;
      size_t epochs;
      double learnRate;
      double learnRateDecay;
      size_t maxBatchSize;
      double dropoutRate;

      nlohmann::json toJson() const;
    };

    Params m_params;
    size_t m_numInputs;
    std::vector<Layer> m_layers;
    bool m_isTrained;
};
