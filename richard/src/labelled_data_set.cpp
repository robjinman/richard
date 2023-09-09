#include <limits>
#include "labelled_data_set.hpp"
#include "exception.hpp"

namespace {

void normalizeSamples(std::vector<Sample>& samples, const Vector& min, const Vector& max) {
  for (auto& sample : samples) {
    for (size_t i = 0; i < sample.data.size(); ++i) {
      if (max[i] > 0.0) {
        sample.data[i] = (sample.data[i] - min[i]) / (max[i] - min[i]);
      }
      else {
        sample.data[i] = 0.0;
      }
    }
  }
}

void computeMinMax(const std::vector<Sample>& samples, Vector& min, Vector& max) {
  TRUE_OR_THROW(!samples.empty(), "Samples vector is empty");

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
}

}

LabelledDataSet::LabelledDataSet(const std::vector<std::string>& labels)
  : m_labels(labels) {

  for (size_t i = 0; i < m_labels.size(); ++i) {
    Vector v(m_labels.size());
    v.zero();
    v[i] = 1.0;
    m_classOutputVectors.insert({m_labels[i], v});
  }
}

LabelledDataSet::~LabelledDataSet() {}

void normalizeTrainingSamples(std::vector<Sample>& samples) {
  TRUE_OR_THROW(!samples.empty(), "Dataset is empty");

  Vector min(samples[0].data.size());
  Vector max(samples[0].data.size());

  min.fill(std::numeric_limits<double>::max());
  max.fill(std::numeric_limits<double>::min());

  computeMinMax(samples, min, max);

  normalizeSamples(samples, min, max);
}

void normalizeTestSamples(std::vector<Sample>& samples, const Vector& trainingMin,
  const Vector& trainingMax) {

  TRUE_OR_THROW(!samples.empty(), "Dataset is empty");

  Vector min = trainingMin;
  Vector max = trainingMax;

  computeMinMax(samples, min, max);

  normalizeSamples(samples, min, max);
}
