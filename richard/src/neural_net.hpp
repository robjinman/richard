#pragma once

#include <vector>
#include <map>
#include <set>
#include "math.hpp"

struct Sample {
  Sample(char label, const Vector& data)
    : label(label)
    , data(data) {}

  char label;
  Vector data;
};

class Dataset {
  public:
    explicit Dataset(const std::vector<char>& labels);

    inline void addSample(char label, const Vector& data);
    inline const Vector& classOutputVector(char label) const;
    inline std::vector<Sample>& samples();
    inline const std::vector<Sample>& samples() const;
    void normalize(const Vector& min, const Vector& max);

  private:
    std::vector<char> m_labels;
    std::map<char, Vector> m_classOutputVectors;
    std::vector<Sample> m_samples;
};

inline void Dataset::addSample(char label, const Vector& data) {
  m_samples.emplace_back(label, data);
}

inline const std::vector<Sample>& Dataset::samples() const {
  return m_samples;
}

inline const Vector& Dataset::classOutputVector(char label) const {
  return m_classOutputVectors.at(label);
}

inline std::vector<Sample>& Dataset::samples() {
  return m_samples;
}

class TrainingData {
  public:
    explicit TrainingData(std::unique_ptr<Dataset> data);

    inline const Dataset& data() const;
    void normalize();
    inline const Vector& min() const;
    inline const Vector& max() const;

  private:
    std::unique_ptr<Dataset> m_data;
    Vector m_min;
    Vector m_max;
};

inline const Dataset& TrainingData::data() const {
  return *m_data;
}

inline const Vector& TrainingData::min() const {
  return m_min;
}

inline const Vector& TrainingData::max() const {
  return m_max;
}

class TestData {
  public:
    explicit TestData(std::unique_ptr<Dataset> data);

    inline const Dataset& data() const;
    void normalize(const Vector& trainingDataMin, const Vector& trainingDataMax);

  private:
    std::unique_ptr<Dataset> m_data;
};

inline const Dataset& TestData::data() const {
  return *m_data;
}

class NeuralNet {
  public:
    struct Results {
      size_t good = 0;
      size_t bad = 0;
      double cost = 0.0;
    };

    explicit NeuralNet(std::initializer_list<size_t> layers);

    size_t inputSize() const;
    void toFile(const TrainingData& trainingData, const std::string& filePath) const;
    void fromFile(const std::string& filePath, Vector& trainingDataMin, Vector& trainingDataMax);
    void train(const TrainingData& data);
    Results test(const TestData& data) const;
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
      std::set<size_t> dropSet;

      Layer(Layer&& mv);
      Layer(Matrix&& weights, Vector&& biases);
      Layer(const Layer& cpy);
      Layer(const Matrix& weights, const Vector& biases);
    };

    size_t m_numInputs;
    std::vector<Layer> m_layers;
};
