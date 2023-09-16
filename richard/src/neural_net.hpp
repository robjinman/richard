#pragma once

#include "math.hpp"

struct HyperParams {
  size_t epochs = 50;
  double learnRate = 0.7;
  double learnRateDecay = 1.0;
  size_t maxBatchSize = 1000;
  double dropoutRate = 0.5;
};

struct NetworkConfig {
  explicit NetworkConfig(const std::vector<size_t>& layers);
  explicit NetworkConfig(std::istream& s);

  void writeToStream(std::ostream& s) const;

  static void printExample(std::ostream& s);
  static NetworkConfig fromFile(const std::string& filePath);

  std::vector<size_t> layers;
  HyperParams params;
};

class LabelledDataSet;

class NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    explicit NeuralNet(const NetworkConfig& config);
    explicit NeuralNet(std::istream& s);

    CostFn costFn() const;
    size_t inputSize() const;
    void writeToStream(std::ostream& s) const;
    void train(LabelledDataSet& data);
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
    NetworkConfig m_config;
    bool m_isTrained;
};
