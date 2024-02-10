#pragma once

#include "application.hpp"
#include <richard/classifier.hpp>
#include <richard/data_details.hpp>
#include <richard/labelled_data_set.hpp>

class Outputter;

namespace richard {

class EventSystem;
class FileSystem;
class PlatformPaths;
class Logger;

class ClassifierEvalApp : public Application {
  public:
    struct Options {
      std::string samplesPath;
      std::string networkFile;
      bool gpuAccelerated;
    };

    ClassifierEvalApp(EventSystem& eventSystem, FileSystem& fileSystem,
      const PlatformPaths& platformPaths, const Options& options, Outputter& outputter,
      Logger& logger);

    std::string name() const override;
    void start() override;

  private:
    EventSystem& m_eventSystem;
    FileSystem& m_fileSystem;
    Outputter& m_outputter;
    Logger& m_logger;
    Options m_opts;
    std::unique_ptr<Classifier> m_classifier;
    std::unique_ptr<DataDetails> m_dataDetails;
    std::unique_ptr<LabelledDataSet> m_dataSet;
};

}
