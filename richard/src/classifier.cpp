#include "classifier.hpp"
#include "labelled_data_set.hpp"
#include "exception.hpp"
#include "data_details.hpp"
#include "util.hpp"
#include "logger.hpp"

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

Classifier::Classifier(const DataDetails& dataDetails, const nlohmann::json& config,
  std::istream& fin, Logger& logger)
  : m_logger(logger)
  , m_isTrained(false) {

  m_neuralNet = createNeuralNet(dataDetails.shape, getOrThrow(config, "network"), fin, m_logger);

  m_isTrained = true;
}

Classifier::Classifier(const DataDetails& dataDetails, const nlohmann::json& config, Logger& logger)
  : m_logger(logger)
  , m_neuralNet(nullptr)
  , m_isTrained(false) {

  m_neuralNet = createNeuralNet(dataDetails.shape, getOrThrow(config, "network"), m_logger);
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
        m_logger.info("1", false);
      }
      else {
        ++results.bad;
        m_logger.info("0", false);
      }

      totalCost += costFn(*actual, expected);
      ++totalSamples;
    }
    samples.clear();
  }
  m_logger.info("");

  results.cost = totalCost / totalSamples;

  return results;
}

void Classifier::abort() {
  m_neuralNet->abort();
}

const nlohmann::json& Classifier::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;

  if (!done) {
    obj["network"] = NeuralNet::exampleConfig();
    done = true;
  }
  
  return obj;
}

