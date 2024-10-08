#include "classifier_training_app.hpp"
#include "outputter.hpp"
#include <richard/stdin_monitor.hpp>
#include <richard/event_system.hpp>
#include <richard/file_system.hpp>
#include <richard/logger.hpp>

namespace richard {

ClassifierTrainingApp::ClassifierTrainingApp(EventSystem& eventSystem, FileSystem& fileSystem,
  const PlatformPaths& platformPaths, const Options& options, Outputter& outputter, Logger& logger)
  : m_outputter(outputter)
  , m_eventSystem(eventSystem)
  , m_fileSystem(fileSystem)
  , m_opts(options) {

  auto stream = m_fileSystem.openFileForReading(m_opts.configFile);
  m_config = Config::fromJson(*stream);

  m_dataDetails = std::make_unique<DataDetails>(m_config.getObject("data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, m_config.getObject("classifier"),
    eventSystem, fileSystem, platformPaths, logger, m_opts.gpuAccelerated);

  auto loader = createDataLoader(m_fileSystem, m_config.getObject("dataLoader"), m_opts.samplesPath,
    *m_dataDetails);

  m_dataSet = std::make_unique<LabelledDataSet>(std::move(loader), m_dataDetails->classLabels);
}

std::string ClassifierTrainingApp::name() const {
  return "Classifier Training";
}

void ClassifierTrainingApp::start() {
  StdinMonitor stdinMonitor;
  stdinMonitor.onKey('q', [this]() { m_classifier->abort(); });

  m_outputter.printLine("Model details");
  auto details = m_classifier->modelDetails();
  for (auto item : details) {
    m_outputter.printLine(STR("> " << item.first << ": " << item.second));
  }
  m_outputter.printSeparator();
  m_outputter.printLine("Richard is gaining power...");

  auto onEpochStarted = [&](const Event& event) {
    const auto& e = dynamic_cast<const EEpochStarted&>(event); 
    m_outputter.printLine(STR("> Epoch " << e.epoch + 1 << "/" << e.epochs));
  };

  auto onSampleProcessed = [&](const Event& event) {
    const auto& e = dynamic_cast<const ESampleProcessed&>(event); 
    m_outputter.printLine(STR("\r  Sample " << e.sample + 1 << "/" << e.samples), false);
  };

  auto onEpochCompleted = [&](const Event& event) {
    const auto& e = dynamic_cast<const EEpochCompleted&>(event); 
    m_outputter.printLine(STR("\r  Cost " << e.cost << std::string(10, ' ')));
  };

  auto hOnEpochStarted = m_eventSystem.listen(hashString("epochStarted"), onEpochStarted);
  auto hOnEpochCompleted = m_eventSystem.listen(hashString("epochCompleted"), onEpochCompleted);
  auto hOnSampleProcessed = m_eventSystem.listen(hashString("sampleProcessed"), onSampleProcessed);

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
  static Config config = []() {
    Config c;
    c.setObject("data", DataDetails::exampleConfig());
    c.setObject("dataLoader", DataLoader::exampleConfig());
    c.setObject("classifier", Classifier::exampleConfig());
    return c;
  }();

  return config;
}

}
