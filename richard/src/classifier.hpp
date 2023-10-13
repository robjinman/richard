#pragma once

#include <nlohmann/json.hpp>
#include "neural_net.hpp"
#include "data_stats.hpp"

class TrainingDataSet;

class Classifier {
  public:
    struct Results {
      size_t good = 0;
      size_t bad = 0;
      double cost = 0.0;
    };

    explicit Classifier(const nlohmann::json& config);
    explicit Classifier(const std::string& filePath);

    void toFile(const std::string& filePath) const;
    void train(TrainingDataSet& trainingData);
    Results test(LabelledDataSet& testData) const;
    std::array<size_t, 2> inputSize() const;
    const std::vector<std::string> classLabels() const;
    const DataStats& trainingDataStats() const;

    // Called from another thread
    void abort();

    static const nlohmann::json& defaultConfig();

  private:
    std::unique_ptr<NeuralNet> m_neuralNet;
    std::vector<std::string> m_classes;
    bool m_isTrained;
    std::unique_ptr<DataStats> m_trainingDataStats;
};
