#include <fstream>
#include <limits>
#include <sstream>
#include <iostream> // TODO
#include "classifier.hpp"

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

Classifier::Classifier(const std::string& filePath)
  : m_isTrained(false)
  , m_trainingSetMin(1)
  , m_trainingSetMax(1) {

  std::ifstream fin(filePath, std::ios::binary);
  m_neuralNet = std::make_unique<NeuralNet>(fin);

  size_t classesStringSize = 0;
  fin.read(reinterpret_cast<char*>(&classesStringSize), sizeof(size_t));
  std::string classesString(classesStringSize, '_');
  fin.read(classesString.data(), classesStringSize);
  std::stringstream ss{classesString};
  std::string label;
  while (std::getline(ss, label, ',')) {
    m_classes.push_back(label);
  }

  size_t numInputs = m_neuralNet->inputSize();

  Vector min(numInputs);
  Vector max(numInputs);

  fin.read(reinterpret_cast<char*>(min.data()), numInputs * sizeof(double));
  fin.read(reinterpret_cast<char*>(max.data()), numInputs * sizeof(double));

  m_trainingSetMin = min;
  m_trainingSetMax = max;
}

Classifier::Classifier(const std::vector<size_t>& layers, const std::vector<std::string>& classes)
  : m_neuralNet(std::make_unique<NeuralNet>(layers))
  , m_classes(classes)
  , m_isTrained(false)
  , m_trainingSetMin(1)
  , m_trainingSetMax(1) {}

size_t Classifier::inputSize() const {
  return m_neuralNet->inputSize();
}

const std::vector<std::string> Classifier::classLabels() const {
  return m_classes;
}

void Classifier::toFile(const std::string& filePath) const {
  ASSERT(m_isTrained);

  std::ofstream fout(filePath, std::ios::binary);

  m_neuralNet->toFile(fout);

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

  fout.write(reinterpret_cast<const char*>(m_trainingSetMin.data()),
    m_trainingSetMin.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_trainingSetMax.data()),
    m_trainingSetMax.size() * sizeof(double));
}

void Classifier::train(const TrainingData& data) {
  m_neuralNet->train(data);
  m_trainingSetMin = data.min();
  m_trainingSetMax = data.max();
  m_isTrained = true;
}

Classifier::Results Classifier::test(const TestData& testData) const {
  ASSERT(m_isTrained);

  Results results;

  const Dataset& data = testData.data();
  const auto& samples = data.samples();

  const auto& costFn = m_neuralNet->costFn();

  double totalCost = 0.0;
  for (const auto& sample : samples) {
    Vector actual = m_neuralNet->evaluate(sample.data);
    Vector expected = data.classOutputVector(sample.label);

    if (outputsMatch(actual, expected)) {
      ++results.good;
      std::cout << "1" << std::flush;
    }
    else {
      ++results.bad;
      std::cout << "0" << std::flush;
    }

    totalCost += costFn(actual, expected);
  }
  std::cout << std::endl;

  results.cost = totalCost / data.samples().size();

  return results;
}

Vector Classifier::trainingSetMin() const {
  ASSERT(m_isTrained);
  return m_trainingSetMin;
}

Vector Classifier::trainingSetMax() const {
  ASSERT(m_isTrained);
  return m_trainingSetMax;
}
