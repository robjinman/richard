#include <limits>
#include "training_data_set.hpp"

TrainingDataSet::TrainingDataSet(std::unique_ptr<DataLoader> loader,
  const std::vector<std::string>& labels, bool computeStatsAndNormalize)
  : LabelledDataSet(labels)
  , m_loader(std::move(loader))
  , m_computeStatsAndNormalize(computeStatsAndNormalize)
  , m_stats(nullptr) {}

void TrainingDataSet::seekToBeginning() {
  m_loader->seekToBeginning();
}

size_t TrainingDataSet::loadSamples(std::vector<Sample>& samples, size_t n) {
  samples.clear();
  size_t numLoaded = m_loader->loadSamples(samples, n);

  if (m_computeStatsAndNormalize) {
    if (m_stats == nullptr && numLoaded > 0) {
      size_t inputSize = samples[0].data.size();
      m_stats = std::make_unique<DataStats>(Vector(inputSize), Vector(inputSize));
      m_stats->min.fill(std::numeric_limits<double>::max());
      m_stats->min.fill(std::numeric_limits<double>::min());
    }

    if (m_stats != nullptr) {
      Vector& min = m_stats->min;
      Vector& max = m_stats->max;

      for (const Sample& sample : samples) {
        for (size_t i = 0; i < sample.data.size(); ++i) {
          double value = sample.data[i];
          if (value < min[i]) {
            min[i] = value;
          }
          if (value > max[i]) {
            max[i] = value;
          }
        }
      }

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
  }

  return numLoaded;
}

const DataStats& TrainingDataSet::stats() const {
  return *m_stats;
}
