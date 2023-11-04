#pragma once

#include "application.hpp"
#include "classifier.hpp"
#include "data_details.hpp"
#include "labelled_data_set.hpp"

class FileSystem;

class ClassifierEvalApp : public Application {
  public:
    struct Options {
      std::string samplesPath;
      std::string networkFile;
    };

    ClassifierEvalApp(FileSystem& fileSystem, const Options& options);

    void start() override;

  private:
    FileSystem& m_fileSystem;
    Options m_opts;
    std::unique_ptr<Classifier> m_classifier;
    std::unique_ptr<DataDetails> m_dataDetails;
    std::unique_ptr<LabelledDataSet> m_dataSet;
};

