#include "classifier_eval_app.hpp"
#include "outputter.hpp"
#include <richard/event_system.hpp>
#include <richard/file_system.hpp>
#include <richard/logger.hpp>

namespace richard {

ClassifierEvalApp::ClassifierEvalApp(EventSystem& eventSystem, FileSystem& fileSystem,
  const PlatformPaths& platformPaths, const Options& options, Outputter& outputter, Logger& logger)
  : m_eventSystem(eventSystem)
  , m_fileSystem(fileSystem)
  , m_outputter(outputter)
  , m_logger(logger)
  , m_opts(options) {

  auto stream = m_fileSystem.openFileForReading(m_opts.networkFile);

  size_t configSize = 0;
  stream->read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  stream->read(reinterpret_cast<char*>(configString.data()), configSize);
  Config config = Config::fromJson(configString);

  m_dataDetails = std::make_unique<DataDetails>(config.getObject("data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, config.getObject("classifier"),
    *stream, eventSystem, fileSystem, platformPaths, m_logger, m_opts.gpuAccelerated);

  auto loader = createDataLoader(m_fileSystem, config.getObject("dataLoader"),
    m_opts.samplesPath, *m_dataDetails);

  m_dataSet = std::make_unique<LabelledDataSet>(std::move(loader), m_dataDetails->classLabels);
}

std::string ClassifierEvalApp::name() const {
  return "Classifier Evaluation";
}

void ClassifierEvalApp::start() {
  Classifier::Results results = m_classifier->test(*m_dataSet);

  for (bool guess : results.guesses) {
    m_outputter.printLine(guess ? "1" : "0", false);
  }
  m_outputter.printLine("");

  m_outputter.printSeparator();

  m_outputter.printLine(STR("Correct classifications: "
    << results.good << "/" << results.good + results.bad));

  m_outputter.printLine(STR("Average cost: " << results.cost));
}

}
