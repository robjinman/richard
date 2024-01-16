#pragma once

#include "neural_net.hpp"
#include <nlohmann/json.hpp>

namespace richard {

class LabelledDataSet;
class DataDetails;
class Logger;
class FileSystem;
class PlatformPaths;

class Classifier {
  public:
    struct Results {
      size_t good = 0;
      size_t bad = 0;
      netfloat_t cost = 0.0;
    };

    Classifier(const DataDetails& dataDetails, const nlohmann::json& config, FileSystem& fileSystem,
      const PlatformPaths& platformPaths, Logger& logger, bool gpuAccelerated);
    Classifier(const DataDetails& dataDetails, const nlohmann::json& config, std::istream& fin,
      FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger,
      bool gpuAccelerated);

    void writeToStream(std::ostream& fout) const;
    void train(LabelledDataSet& trainingData);
    Results test(LabelledDataSet& testData) const;

    // Called from another thread
    void abort();

    static const nlohmann::json& exampleConfig();

  private:
    Logger& m_logger;
    std::unique_ptr<NeuralNet> m_neuralNet;
    bool m_isTrained;
};

}
