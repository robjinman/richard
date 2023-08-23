#pragma once

#include <vector>
#include <map>
#include "math.hpp"

class TrainingData {
  public:
    struct Sample {
      Sample(char label, const Vector& data)
        : label(label)
        , data(data) {}

      char label;
      Vector data;
    };

    TrainingData(const std::vector<char>& labels);

    inline void addSample(char label, const Vector& data);
    inline const std::vector<Sample>& data() const;
    inline const Vector& classOutputVector(char label) const;
    void normalize();

  private:
    std::vector<char> m_labels;
    std::map<char, Vector> m_classOutputVectors;
    std::vector<Sample> m_samples;
};

inline void TrainingData::addSample(char label, const Vector& data) {
  m_samples.emplace_back(label, data);
}

inline const std::vector<TrainingData::Sample>& TrainingData::data() const {
  return m_samples;
}

inline const Vector& TrainingData::classOutputVector(char label) const {
  return m_classOutputVectors.at(label);
}

class NeuralNet {
  public:
    NeuralNet(std::initializer_list<size_t> layers);

    void train(const TrainingData& data);
    Vector evaluate(const Vector& inputs) const;

    // For testing
    void setWeights(const std::vector<Matrix>& W);
    void setBiases(const std::vector<Vector>& B);

  private:
    void feedForward(const Vector& x);
    void updateLayer(size_t layerIdx, const Vector& delta, const Vector& x);

    struct Layer {
      Matrix weights;
      Vector biases;
      Vector Z;

      //Layer(Layer&& mv);
      //Layer(Matrix&& weights, Vector&& biases);
      Layer(const Layer& cpy);
      Layer(const Matrix& weights, const Vector& biases);
    };

    std::vector<Layer> m_layers;
};
