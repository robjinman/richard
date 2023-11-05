#include <iostream>
#include "classifier_training_app.hpp"
#include "stdin_monitor.hpp"
#include "util.hpp"
#include "file_system.hpp"
#include "logger.hpp"

ClassifierTrainingApp::ClassifierTrainingApp(FileSystem& fileSystem, const Options& options,
  Logger& logger)
  : m_logger(logger)
  , m_fileSystem(fileSystem)
  , m_opts(options) {

  auto fin = m_fileSystem.openFileForReading(options.configFile);
  m_config = nlohmann::json::parse(*fin);

  m_dataDetails = std::make_unique<DataDetails>(getOrThrow(m_config, "data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, getOrThrow(m_config, "classifier"),
    m_logger);

  m_dataSet = createDataSet(m_fileSystem, m_opts.samplesPath, *m_dataDetails);
}

void ClassifierTrainingApp::start() {
  StdinMonitor stdinMonitor;
  stdinMonitor.onKey('q', [this]() { m_classifier->abort(); });

  m_logger.info("Training classifier");

  m_classifier->train(*m_dataSet);

  saveStateToFile();
}

void ClassifierTrainingApp::saveStateToFile() const {
  auto fout = m_fileSystem.openFileForWriting(m_opts.networkFile);

  std::string configString = m_config.dump();
  size_t configSize = configString.size();
  fout->write(reinterpret_cast<char*>(&configSize), sizeof(size_t));
  fout->write(configString.c_str(), configSize);

  m_classifier->writeToStream(*fout);
}

const nlohmann::json& ClassifierTrainingApp::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;

  if (!done) {
    obj["data"] = DataDetails::exampleConfig();
    obj["classifier"] = Classifier::exampleConfig();

    done = true;
  }

  return obj;
}

