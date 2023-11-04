#pragma once

#include <nlohmann/json.hpp>
#include "neural_net.hpp"

class LabelledDataSet;
class DataDetails;

class Classifier {
  public:
    struct Results {
      size_t good = 0;
      size_t bad = 0;
      double cost = 0.0;
    };

    explicit Classifier(const DataDetails& dataDetails, const nlohmann::json& config);
    Classifier(const DataDetails& dataDetails, const nlohmann::json& config, std::istream& fin);

    void writeToStream(std::ostream& fout) const;
    void train(LabelledDataSet& trainingData);
    Results test(LabelledDataSet& testData) const;

    // Called from another thread
    void abort();

    static const nlohmann::json& exampleConfig();

  private:
    std::unique_ptr<NeuralNet> m_neuralNet;
    bool m_isTrained;
};

