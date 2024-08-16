#include "richard/classifier.hpp"
#include "richard/labelled_data_set.hpp"
#include "richard/exception.hpp"
#include "richard/data_details.hpp"
#include "richard/utils.hpp"
#include "richard/logger.hpp"
#include "richard/cpu/cpu_neural_net.hpp"
#include "richard/gpu/gpu_neural_net.hpp"
#include <limits>
#include <future>

namespace richard {
namespace {

bool outputsMatch(const Vector& x, const Vector& y) {
  auto largestComponent = [](const Vector& v) {
    netfloat_t largest = std::numeric_limits<netfloat_t>::min();
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

Classifier::Classifier(const DataDetails& dataDetails, const Config& config, std::istream& stream,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger, bool gpuAccelerated)
  : m_eventSystem(eventSystem)
  , m_isTrained(false) {

  if (gpuAccelerated) {
    m_neuralNet = gpu::createNeuralNet(dataDetails.shape, config.getObject("network"), stream,
      m_eventSystem, fileSystem, platformPaths, logger);
  }
  else {
    m_neuralNet = cpu::createNeuralNet(dataDetails.shape, config.getObject("network"), stream,
      m_eventSystem);
  }

  m_isTrained = true;
}

Classifier::Classifier(const DataDetails& dataDetails, const Config& config,
  EventSystem& eventSystem, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  Logger& logger, bool gpuAccelerated)
  : m_eventSystem(eventSystem)
  , m_neuralNet(nullptr)
  , m_isTrained(false) {

  if (gpuAccelerated) {
    m_neuralNet = gpu::createNeuralNet(dataDetails.shape, config.getObject("network"),
      m_eventSystem, fileSystem, platformPaths, logger);
  }
  else {
    m_neuralNet = cpu::createNeuralNet(dataDetails.shape, config.getObject("network"),
      m_eventSystem);
  }
}

ModelDetails Classifier::modelDetails() const {
  return m_neuralNet->modelDetails();
}

void Classifier::writeToStream(std::ostream& stream) const {
  ASSERT_MSG(m_isTrained, "Classifier not trained");

  m_neuralNet->writeToStream(stream);
}

void Classifier::train(LabelledDataSet& data) {
  m_neuralNet->train(data);
  m_isTrained = true;
}

Classifier::Results Classifier::test(LabelledDataSet& testData) const {
  ASSERT_MSG(m_isTrained, "Classifier not trained");

  Results results;

  const auto& costFn = m_neuralNet->costFn();

  [[maybe_unused]] size_t netInputSize = calcProduct(m_neuralNet->inputSize());

  auto pendingSamples = std::async([&]() { return testData.loadSamples(); });
  std::vector<Sample> samples = pendingSamples.get();

  size_t totalSamples = 0;
  netfloat_t totalCost = 0.0;
  while (samples.size() > 0) {
    pendingSamples = std::async([&]() { return testData.loadSamples(); });

    for (const auto& sample : samples) {
      DBG_ASSERT_MSG(sample.data.size() == netInputSize,
        "Expected sample of size " << netInputSize << ", got " << sample.data.size());

      Vector actual = m_neuralNet->evaluate(sample.data);
      Vector expected = testData.classOutputVector(sample.label);

      if (outputsMatch(actual, expected)) {
        ++results.good;
        results.guesses.push_back(true);
      }
      else {
        ++results.bad;
        results.guesses.push_back(false);
      }

      totalCost += costFn(actual, expected);
      ++totalSamples;
    }

    samples = pendingSamples.get();
  }

  results.cost = totalCost / totalSamples;

  return results;
}

void Classifier::abort() {
  m_neuralNet->abort();
}

const Config& Classifier::exampleConfig() {
  static Config config = []() {
    Config c;
    c.setObject("network", NeuralNet::exampleConfig());
    return c;
  }();
  
  return config;
}

}
