#include <iostream> // TODO
#include "classifier.hpp"
#include "labelled_data_set.hpp"
#include "exception.hpp"
#include "data_details.hpp"
#include "util.hpp"

namespace {

bool outputsMatch(const Vector& x, const Vector& y) {
  auto largestComponent = [](const Vector& v) {
    double largest = std::numeric_limits<double>::min();
    size_t largestIdx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i] > largest) {
        largest = v[i];
        largestIdx = i;
      }
    }
    return largestIdx;
  };

  return largestComponent(x) == largestComponent(y);
}

}

const nlohmann::json& Classifier::defaultConfig() {
  static nlohmann::json config;
  config["network"] = NeuralNet::defaultConfig();
  return config;
}

Classifier::Classifier(const DataDetails& dataDetails, const nlohmann::json& config,
  std::istream& fin)
  : m_isTrained(false) {

  m_neuralNet = createNeuralNet(dataDetails.shape, getOrThrow(config, "network"), fin);

  m_isTrained = true;
}

Classifier::Classifier(const DataDetails& dataDetails, const nlohmann::json& config)
  : m_neuralNet(nullptr)
  , m_isTrained(false) {

  m_neuralNet = createNeuralNet(dataDetails.shape, getOrThrow(config, "network"));
}

void Classifier::writeToStream(std::ostream& fout) const {
  TRUE_OR_THROW(m_isTrained, "Classifier not trained");

  m_neuralNet->writeToStream(fout);
}

void Classifier::train(LabelledDataSet& data) {
  m_neuralNet->train(data);
  m_isTrained = true;
}

Classifier::Results Classifier::test(LabelledDataSet& testData) const {
  TRUE_OR_THROW(m_isTrained, "Classifier not trained");

  const size_t N = 500; // TODO

  Results results;

  std::vector<Sample> samples;
  const auto& costFn = m_neuralNet->costFn();

  auto inputSz = m_neuralNet->inputSize();
  size_t netInputSize = inputSz[0] * inputSz[1] * inputSz[2];

  size_t totalSamples = 0;
  double totalCost = 0.0;
  while (testData.loadSamples(samples, N) > 0) {
    for (const auto& sample : samples) {
      TRUE_OR_THROW(sample.data.size() == netInputSize,
        "Expected sample of size " << netInputSize << ", got " << sample.data.size());

      std::unique_ptr<Vector> actual = m_neuralNet->evaluate(sample.data);
      Vector expected = testData.classOutputVector(sample.label);

      if (outputsMatch(*actual, expected)) {
        ++results.good;
        std::cout << "1" << std::flush;
      }
      else {
        ++results.bad;
        std::cout << "0" << std::flush;
      }

      totalCost += costFn(*actual, expected);
      ++totalSamples;
    }
    samples.clear();
  }
  std::cout << std::endl;

  results.cost = totalCost / totalSamples;

  return results;
}

void Classifier::abort() {
  m_neuralNet->abort();
}

