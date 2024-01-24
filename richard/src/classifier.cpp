#include "classifier.hpp"
#include "labelled_data_set.hpp"
#include "exception.hpp"
#include "data_details.hpp"
#include "utils.hpp"
#include "logger.hpp"
#include "cpu/cpu_neural_net.hpp"
#include "gpu/gpu_neural_net.hpp"
#include <limits>

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
  FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger, bool gpuAccelerated)
  : m_logger(logger)
  , m_isTrained(false) {

  if (gpuAccelerated) {
    m_neuralNet = gpu::createNeuralNet(dataDetails.shape, config.getObject("network"), stream,
      fileSystem, platformPaths, m_logger);
  }
  else {
    m_neuralNet = cpu::createNeuralNet(dataDetails.shape, config.getObject("network"), stream,
      m_logger);
  }

  m_isTrained = true;
}

Classifier::Classifier(const DataDetails& dataDetails, const Config& config,
  FileSystem& fileSystem, const PlatformPaths& platformPaths, Logger& logger, bool gpuAccelerated)
  : m_logger(logger)
  , m_neuralNet(nullptr)
  , m_isTrained(false) {

  if (gpuAccelerated) {
    m_neuralNet = gpu::createNeuralNet(dataDetails.shape, config.getObject("network"), fileSystem,
      platformPaths, m_logger);
  }
  else {
    m_neuralNet = cpu::createNeuralNet(dataDetails.shape, config.getObject("network"), m_logger);
  }
}

void Classifier::writeToStream(std::ostream& fout) const {
  ASSERT_MSG(m_isTrained, "Classifier not trained");

  m_neuralNet->writeToStream(fout);
}

void Classifier::train(LabelledDataSet& data) {
  m_neuralNet->train(data);
  m_isTrained = true;
}

Classifier::Results Classifier::test(LabelledDataSet& testData) const {
  ASSERT_MSG(m_isTrained, "Classifier not trained");

  Results results;

  std::vector<Sample> samples;
  const auto& costFn = m_neuralNet->costFn();

  [[maybe_unused]] size_t netInputSize = calcProduct(m_neuralNet->inputSize());

  size_t totalSamples = 0;
  netfloat_t totalCost = 0.0;
  while (testData.loadSamples(samples) > 0) {
    for (const auto& sample : samples) {
      DBG_ASSERT_MSG(sample.data.size() == netInputSize,
        "Expected sample of size " << netInputSize << ", got " << sample.data.size());

      Vector actual = m_neuralNet->evaluate(sample.data);
      Vector expected = testData.classOutputVector(sample.label);

      if (outputsMatch(actual, expected)) {
        ++results.good;
        m_logger.info("1", false);
      }
      else {
        ++results.bad;
        m_logger.info("0", false);
      }

      totalCost += costFn(actual, expected);
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

const Config& Classifier::exampleConfig() {
  static Config obj;
  static bool done = false;

  if (!done) {
    obj.setObject("network", NeuralNet::exampleConfig());

    done = true;
  }
  
  return obj;
}

}
