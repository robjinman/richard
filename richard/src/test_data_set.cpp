#include "test_data_set.hpp"

TestDataSet::TestDataSet(std::unique_ptr<DataLoader> loader, const std::vector<std::string>& labels)
  : LabelledDataSet(labels)
  , m_loader(std::move(loader))
  , m_normalize(false)
  , m_trainingDataStats(Vector(1), Vector(1)) {}

void TestDataSet::seekToBeginning() {
  m_loader->seekToBeginning();
}

void TestDataSet::normalize(const DataStats& trainingDataStats) {
  m_trainingDataStats = trainingDataStats;
  m_normalize = true;
}

size_t TestDataSet::loadSamples(std::vector<Sample>& samples, size_t n) {
  samples.clear();
  size_t numLoaded = m_loader->loadSamples(samples, n);

  if (m_normalize) {
    const Vector& min = m_trainingDataStats.min;
    const Vector& max = m_trainingDataStats.max;

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

  return numLoaded;
}
