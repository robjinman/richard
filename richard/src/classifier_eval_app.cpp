#include "classifier_eval_app.hpp"
#include "util.hpp"
#include "file_system.hpp"
#include "logger.hpp"
#include <iostream>

ClassifierEvalApp::ClassifierEvalApp(FileSystem& fileSystem, const Options& options, Logger& logger)
  : m_logger(logger)
  , m_fileSystem(fileSystem)
  , m_opts(options) {

  auto fin = m_fileSystem.openFileForReading(options.networkFile);

  size_t configSize = 0;
  fin->read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  fin->read(reinterpret_cast<char*>(configString.data()), configSize);
  nlohmann::json config = nlohmann::json::parse(configString);

  m_dataDetails = std::make_unique<DataDetails>(getOrThrow(config, "data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, getOrThrow(config, "classifier"),
    *fin, m_logger);

  auto loader = createDataLoader(m_fileSystem, getOrThrow(config, "dataLoader"),
    m_opts.samplesPath, *m_dataDetails);

  m_dataSet = std::make_unique<LabelledDataSet>(std::move(loader), m_dataDetails->classLabels);
}

void ClassifierEvalApp::start() {
  m_logger.info("Testing classifier");

  Classifier::Results results = m_classifier->test(*m_dataSet);

  m_logger.info(STR("Correct classifications: "
    << results.good << "/" << results.good + results.bad));

  m_logger.info(STR("Average cost: " << results.cost));
}

