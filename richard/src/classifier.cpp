#include <fstream>
#include <limits>
#include <sstream>
#include <iostream> // TODO
#include "classifier.hpp"
#include "exception.hpp"
#include "training_data_set.hpp"
#include "test_data_set.hpp"

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
  config["classes"] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
  config["network"] = NeuralNet::defaultConfig();
  return config;
}

Classifier::Classifier(const std::string& filePath)
  : m_isTrained(false)
  , m_trainingDataStats(nullptr) {

  std::ifstream fin(filePath, std::ios::binary);
  m_neuralNet = createNeuralNet(fin);

  size_t classesStringSize = 0;
  fin.read(reinterpret_cast<char*>(&classesStringSize), sizeof(size_t));
  std::string classesString(classesStringSize, '_');
  fin.read(classesString.data(), classesStringSize);
  std::stringstream ss{classesString};
  std::string label;
  while (std::getline(ss, label, ',')) {
    m_classes.push_back(label);
  }

  uint32_t hasStats = false;
  fin.read(reinterpret_cast<char*>(&hasStats), sizeof(uint32_t));

  if (hasStats) {
    std::array<size_t, 2> netInputSize = m_neuralNet->inputSize();
    size_t numInputs = netInputSize[0] * netInputSize[1];

    Vector min(numInputs);
    Vector max(numInputs);

    fin.read(reinterpret_cast<char*>(min.data()), numInputs * sizeof(double));
    fin.read(reinterpret_cast<char*>(max.data()), numInputs * sizeof(double));

    m_trainingDataStats = std::make_unique<DataStats>(min, max);
  }

  m_isTrained = true;
}

Classifier::Classifier(const nlohmann::json& config)
  : m_neuralNet(nullptr)
  , m_isTrained(false)
  , m_trainingDataStats(nullptr) {

  auto classesJson = getOrThrow(config, "classes");

  for (auto& label : classesJson) {
    m_classes.push_back(label.get<std::string>());
  }

  m_neuralNet = createNeuralNet(getOrThrow(config, "network"));
}

std::array<size_t, 2> Classifier::inputSize() const {
  return m_neuralNet->inputSize();
}

const std::vector<std::string> Classifier::classLabels() const {
  return m_classes;
}

void Classifier::toFile(const std::string& filePath) const {
  TRUE_OR_THROW(m_isTrained, "Classifier not trained");

  std::ofstream fout(filePath, std::ios::binary);

  m_neuralNet->writeToStream(fout);

  std::stringstream ss;
  for (size_t i = 0; i < m_classes.size(); ++i) {
    ss << m_classes[i];
    if (i + 1 < m_classes.size()) {
      ss << ",";
    }
  }
  size_t n = ss.str().size();
  fout.write(reinterpret_cast<char*>(&n), sizeof(n));
  fout.write(ss.str().c_str(), n);

  uint32_t hasStats = m_trainingDataStats != nullptr;
  fout.write(reinterpret_cast<const char*>(&hasStats), sizeof(uint32_t));

  if (hasStats) {
    fout.write(reinterpret_cast<const char*>(m_trainingDataStats->min.data()),
      m_trainingDataStats->min.size() * sizeof(double));
    fout.write(reinterpret_cast<const char*>(m_trainingDataStats->max.data()),
      m_trainingDataStats->max.size() * sizeof(double));
  }
}

void Classifier::train(TrainingDataSet& data) {
  m_neuralNet->train(data);

  if (data.hasStats()) {
    m_trainingDataStats = std::make_unique<DataStats>(data.stats());
  }

  m_isTrained = true;
}

Classifier::Results Classifier::test(LabelledDataSet& testData) const {
  TRUE_OR_THROW(m_isTrained, "Classifier not trained");

  const size_t N = 500; // TODO

  Results results;

  std::vector<Sample> samples;
  const auto& costFn = m_neuralNet->costFn();

  size_t totalSamples = 0;
  double totalCost = 0.0;
  while (testData.loadSamples(samples, N) > 0) {
    for (const auto& sample : samples) {
      size_t netInputSize = m_neuralNet->inputSize()[0] * m_neuralNet->inputSize()[1];
      TRUE_OR_THROW(sample.data.size() == netInputSize,
        "Expected sample of size " << netInputSize << ", got " << sample.data.size());

      Vector actual = m_neuralNet->evaluate(sample.data);
      Vector expected = testData.classOutputVector(sample.label);

      if (outputsMatch(actual, expected)) {
        ++results.good;
        std::cout << "1" << std::flush;
      }
      else {
        ++results.bad;
        std::cout << "0" << std::flush;
      }

      totalCost += costFn(actual, expected);
      ++totalSamples;
    }
    samples.clear();
  }
  std::cout << std::endl;

  results.cost = totalCost / totalSamples;

  return results;
}

const DataStats& Classifier::trainingDataStats() const {
  TRUE_OR_THROW(m_isTrained, "Classifier not trained");
  TRUE_OR_THROW(m_trainingDataStats != nullptr, "No stats for training set");
  return *m_trainingDataStats;
}

void Classifier::abort() {
  m_neuralNet->abort();
}
