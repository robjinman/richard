#pragma once

#include "neural_net.hpp"

class Classifier {
  public:
    struct Results {
      size_t good = 0;
      size_t bad = 0;
      double cost = 0.0;
    };

    explicit Classifier(const NetworkConfig& config, const std::vector<std::string>& classes);
    explicit Classifier(const std::string& filePath);

    void toFile(const std::string& filePath) const;
    void train(LabelledDataSet& trainingData);
    Results test(LabelledDataSet& testData) const;
    size_t inputSize() const;
    const std::vector<std::string> classLabels() const;

  private:
    std::unique_ptr<NeuralNet> m_neuralNet;
    std::vector<std::string> m_classes;
    bool m_isTrained;
    Vector m_trainingSetMin;
    Vector m_trainingSetMax;
};
