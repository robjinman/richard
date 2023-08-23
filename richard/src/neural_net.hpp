#pragma once

#include <vector>
#include <map>
#include "math.hpp"

class TrainingData {
  public:
    struct Sample {
      char label;
      Vector data;
    };

    TrainingData(const std::vector<char>& labels);

    inline void addSample(char label, const Vector& data);
    inline const std::vector<Sample>& data() const;
    inline const Vector& classOutputVector(char label) const;

  private:
    std::vector<char> m_labels;
    std::map<char, Vector> m_classOutputVectors;
    std::vector<Sample> m_samples;
};

inline void TrainingData::addSample(char label, const Vector& data) {
  m_samples.push_back(Sample{label, data});
}

inline const std::vector<TrainingData::Sample>& TrainingData::data() const {
  return m_samples;
}

inline const Vector& TrainingData::classOutputVector(char label) const {
  return m_classOutputVectors.at(label);
}

struct Layer {
  Matrix weights;
  Vector biases;
  Vector Z;

  Layer(Layer&& mv);
  Layer(Matrix&& weights, Vector&& biases);
};

class NeuralNet {
  public:
    NeuralNet(size_t inputs, std::initializer_list<size_t> layers, size_t outputs);

    void train(const TrainingData& data);
    Vector evaluate(const Vector& inputs) const;

  private:
    void feedForward(const Vector& x);
    void updateLayer(size_t layerIdx, const Vector& delta, const Vector& x);

    Vector m_inputs;
    std::vector<Layer> m_layers;
};
