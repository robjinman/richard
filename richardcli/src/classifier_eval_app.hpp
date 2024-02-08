#pragma once

#include "application.hpp"
#include <richard/classifier.hpp>
#include <richard/data_details.hpp>
#include <richard/labelled_data_set.hpp>

namespace richard {

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

    ClassifierEvalApp(FileSystem& fileSystem, const PlatformPaths& platformPaths,
      const Options& options, Logger& logger);

    std::string name() const override;
    void start() override;

  private:
    Logger& m_logger;
    FileSystem& m_fileSystem;
    Options m_opts;
    std::unique_ptr<Classifier> m_classifier;
    std::unique_ptr<DataDetails> m_dataDetails;
    std::unique_ptr<LabelledDataSet> m_dataSet;
};

}
