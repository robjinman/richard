#include "classifier_eval_app.hpp"
#include <richard/utils.hpp>
#include <richard/file_system.hpp>
#include <richard/logger.hpp>

namespace richard {

ClassifierEvalApp::ClassifierEvalApp(FileSystem& fileSystem, const PlatformPaths& platformPaths,
  const Options& options, Logger& logger)
  : m_logger(logger)
  , m_fileSystem(fileSystem)
  , m_opts(options) {

  auto stream = m_fileSystem.openFileForReading(m_opts.networkFile);

  size_t configSize = 0;
  stream->read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  stream->read(reinterpret_cast<char*>(configString.data()), configSize);
  Config config = Config::fromJson(configString);

  m_dataDetails = std::make_unique<DataDetails>(config.getObject("data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, config.getObject("classifier"),
    *stream, fileSystem, platformPaths, m_logger, m_opts.gpuAccelerated);

  auto loader = createDataLoader(m_fileSystem, config.getObject("dataLoader"),
    m_opts.samplesPath, *m_dataDetails);

  m_dataSet = std::make_unique<LabelledDataSet>(std::move(loader), m_dataDetails->classLabels);
}

std::string ClassifierEvalApp::name() const {
  return "Classifier Evaluation";
}

void ClassifierEvalApp::start() {
  m_logger.info("Testing classifier");

  Classifier::Results results = m_classifier->test(*m_dataSet);

  m_logger.info(STR("Correct classifications: "
    << results.good << "/" << results.good + results.bad));

  m_logger.info(STR("Average cost: " << results.cost));
}

}
