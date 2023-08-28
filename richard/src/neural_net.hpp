#pragma once

#include <vector>
#include <set>
#include "dataset.hpp"

class NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    explicit NeuralNet(std::vector<size_t> layers);
    explicit NeuralNet(std::istream& s);

    CostFn costFn() const;
    size_t inputSize() const;
    void toFile(std::ostream& s) const;
    void train(const TrainingData& data);
    Vector evaluate(const Vector& inputs) const;

    // For unit tests
    void setWeights(const std::vector<Matrix>& W);
    void setBiases(const std::vector<Vector>& B);

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

    size_t m_numInputs;
    std::vector<Layer> m_layers;
    bool m_isTrained;
};
