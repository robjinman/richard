#include <iostream>
#include "classifier_eval_app.hpp"
#include "util.hpp"

ClassifierEvalApp::ClassifierEvalApp(const Options& options)
  : m_opts(options) {

  std::ifstream fin(options.networkFile);

  size_t configSize = 0;
  fin.read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  fin.read(reinterpret_cast<char*>(configString.data()), configSize);
  nlohmann::json config = nlohmann::json::parse(configString);

  m_dataDetails = std::make_unique<DataDetails>(getOrThrow(config, "data"));
  m_classifier = std::make_unique<Classifier>(*m_dataDetails, getOrThrow(config, "classifier"),
    fin);

  m_dataSet = createDataSet(m_opts.samplesPath, *m_dataDetails);
}

void ClassifierEvalApp::start() {
  std::cout << "Testing classifier" << std::endl;

  Classifier::Results results = m_classifier->test(*m_dataSet);

  std::cout << "Correct classifications: "
    << results.good << "/" << results.good + results.bad << std::endl;

  std::cout << "Average cost: " << results.cost << std::endl;
}

