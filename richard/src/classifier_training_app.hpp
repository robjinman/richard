#pragma once

#include <nlohmann/json.hpp>
#include "application.hpp"
#include "data_details.hpp"
#include "classifier.hpp"
#include "labelled_data_set.hpp"

class FileSystem;

class ClassifierTrainingApp : public Application {
  public:
    struct Options {
      std::string samplesPath;
      std::string configFile;
      std::string networkFile;
    };

    ClassifierTrainingApp(FileSystem& fileSystem, const Options& options);

    void start() override;

    static const nlohmann::json& exampleConfig();

  private:
    FileSystem& m_fileSystem;
    Options m_opts;
    nlohmann::json m_config;
    std::unique_ptr<Classifier> m_classifier;
    std::unique_ptr<DataDetails> m_dataDetails;
    std::unique_ptr<LabelledDataSet> m_dataSet;

    void saveStateToFile() const;
};

