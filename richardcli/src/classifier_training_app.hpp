#pragma once

#include "application.hpp"
#include <richard/data_details.hpp>
#include <richard/classifier.hpp>
#include <richard/labelled_data_set.hpp>
#include <richard/config.hpp>

class Outputter;

namespace richard {

class FileSystem;
class Logger;

class ClassifierTrainingApp : public Application {
  public:
    struct Options {
      std::string samplesPath;
      std::string configFile;
      std::string networkFile;
      bool gpuAccelerated;
    };

    ClassifierTrainingApp(EventSystem& eventSystem, FileSystem& fileSystem,
      const PlatformPaths& platformPaths, const Options& options, Outputter& outputter,
      Logger& logger);

    std::string name() const override;
    void start() override;

    static const Config& exampleConfig();

  private:
    void saveStateToFile() const;

    Outputter& m_outputter;
    EventSystem& m_eventSystem;
    FileSystem& m_fileSystem;
    Options m_opts;
    Config m_config;
    std::unique_ptr<Classifier> m_classifier;
    std::unique_ptr<DataDetails> m_dataDetails;
    std::unique_ptr<LabelledDataSet> m_dataSet;
};

}
