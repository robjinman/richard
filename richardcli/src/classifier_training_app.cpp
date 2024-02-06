#include "classifier_training_app.hpp"
#include <richard/stdin_monitor.hpp>
#include <richard/utils.hpp>
#include <richard/file_system.hpp>
#include <richard/logger.hpp>
#include <iostream>

namespace richard {

ClassifierTrainingApp::ClassifierTrainingApp(FileSystem& fileSystem,
  const PlatformPaths& platformPaths, const Options& options, Logger& logger)
  : m_logger(logger)
  , m_fileSystem(fileSystem)
  , m_opts(options) {

  auto stream = m_fileSystem.openFileForReading(m_opts.configFile);
  m_config = Config::fromJson(*stream);

  m_dataDetails = std::make_unique<DataDetails>(m_config.getObject("data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, m_config.getObject("classifier"),
    fileSystem, platformPaths, m_logger, m_opts.gpuAccelerated);

  auto loader = createDataLoader(m_fileSystem, m_config.getObject("dataLoader"), m_opts.samplesPath,
    *m_dataDetails);

  m_dataSet = std::make_unique<LabelledDataSet>(std::move(loader), m_dataDetails->classLabels);
}

void ClassifierTrainingApp::start() {
  StdinMonitor stdinMonitor;
  stdinMonitor.onKey('q', [this]() { m_classifier->abort(); });

  m_logger.info("Training classifier");

  m_classifier->train(*m_dataSet);

  saveStateToFile();
}

void ClassifierTrainingApp::saveStateToFile() const {
  auto stream = m_fileSystem.openFileForWriting(m_opts.networkFile);

  std::string configString = m_config.dump();
  size_t configSize = configString.size();
  stream->write(reinterpret_cast<char*>(&configSize), sizeof(size_t));
  stream->write(configString.c_str(), configSize);

  m_classifier->writeToStream(*stream);
  stream->flush();
}

const Config& ClassifierTrainingApp::exampleConfig() {
  static Config obj;
  static bool done = false;

  if (!done) {
    obj.setObject("data", DataDetails::exampleConfig());
    obj.setObject("dataLoader", DataLoader::exampleConfig());
    obj.setObject("classifier", Classifier::exampleConfig());

    done = true;
  }

  return obj;
}

}
