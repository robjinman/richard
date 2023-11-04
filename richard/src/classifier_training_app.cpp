#include <iostream>
#include "classifier_training_app.hpp"
#include "stdin_monitor.hpp"
#include "util.hpp"

ClassifierTrainingApp::ClassifierTrainingApp(const Options& options)
  : m_opts(options) {

  std::ifstream fin(options.configFile);
  m_config = nlohmann::json::parse(fin);

  m_dataDetails = std::make_unique<DataDetails>(getOrThrow(m_config, "data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, getOrThrow(m_config, "classifier"));

  m_dataSet = createDataSet(m_opts.samplesPath, *m_dataDetails);
}

void ClassifierTrainingApp::start() {
  StdinMonitor stdinMonitor;
  stdinMonitor.onKey('q', [this]() { m_classifier->abort(); });

  std::cout << "Training classifier" << std::endl;

  m_classifier->train(*m_dataSet);

  saveStateToFile();
}

void ClassifierTrainingApp::saveStateToFile() const {
  std::ofstream fout(m_opts.networkFile, std::ios::binary);

  std::string configString = m_config.dump();
  size_t configSize = configString.size();
  fout.write(reinterpret_cast<char*>(&configSize), sizeof(size_t));
  fout.write(configString.c_str(), configSize);

  m_classifier->writeToStream(fout);
}

