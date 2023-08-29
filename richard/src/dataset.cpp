#include <limits>
#include "dataset.hpp"
#include "exception.hpp"

Dataset::Dataset(const std::vector<std::string>& labels) : m_labels(labels) {
  for (size_t i = 0; i < m_labels.size(); ++i) {
    Vector v(m_labels.size());
    v.zero();
    v[i] = 1.0;
    m_classOutputVectors.insert({m_labels[i], v});
  }
}

void Dataset::normalize(const Vector& min, const Vector& max) {
  for (auto& sample : m_samples) {
    for (size_t i = 0; i < sample.data.size(); ++i) {
      sample.data[i] = (sample.data[i] - min[i]) / (max[i] - min[i]);
    }
  }
}

TrainingData::TrainingData(std::unique_ptr<Dataset> data)
  : m_data(std::move(data))
  , m_min(1)
  , m_max(1) {

  TRUE_OR_THROW(!m_data->samples().empty(), "Dataset is empty");

  auto& samples = m_data->samples();

  m_min = Vector(samples[0].data.size());
  m_max = Vector(samples[0].data.size());

  m_min.fill(std::numeric_limits<double>::max());
  m_max.fill(std::numeric_limits<double>::min());

  for (auto& sample : samples) {
    //sample.data.normalize();

    for (size_t i = 0; i < sample.data.size(); ++i) {
      if (sample.data[i] < m_min[i]) {
        m_min[i] = sample.data[i];
      }
      if (sample.data[i] > m_max[i]) {
        m_max[i] = sample.data[i];
      }
    }
  }
}

void TrainingData::normalize() {
  m_data->normalize(m_min, m_max);
}

TestData::TestData(std::unique_ptr<Dataset> data)
  : m_data(std::move(data)) {}

void TestData::normalize(const Vector& trainingMin, const Vector& trainingMax) {
  TRUE_OR_THROW(!m_data->samples().empty(), "Dataset is empty");

  Vector min = trainingMin;
  Vector max = trainingMax;

  auto& samples = m_data->samples();

  for (auto& sample : samples) {
    for (size_t i = 0; i < sample.data.size(); ++i) {
      if (sample.data[i] < min[i]) {
        min[i] = sample.data[i];
      }
      if (sample.data[i] > max[i]) {
        max[i] = sample.data[i];
      }
    }
  }

  m_data->normalize(min, max);
}