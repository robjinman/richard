#include <sstream>
#include <limits>
#include "csv_data_set.hpp"
#include "exception.hpp"

// Load training data from csv file
//
// Each line is a label followed by data values.
// E.g.
//
// b,23.1,45.5
// a,44.0,52.1
// c,11.9,92.4
// ...
CsvDataSet::CsvDataSet(const std::string& filePath, size_t inputSize,
  const std::vector<std::string>& labels)
  : LabelledDataSet(labels)
  , m_inputSize(inputSize)
  , m_fin(filePath)
  , m_stats(nullptr) {}

const DataStats& CsvDataSet::stats() const {
  return *m_stats;
}

void CsvDataSet::seekToBeginning() {
  m_fin.seekg(0);
}

size_t CsvDataSet::loadSamples(std::vector<Sample>& samples, size_t N) {
  size_t numSamples = 0;
  std::string line;
  while (std::getline(m_fin, line)) {
    std::stringstream ss{line};
    std::string label = "_";
    Vector sample(m_inputSize);

    if (m_stats == nullptr) {
      m_stats = std::make_unique<DataStats>(Vector(m_inputSize), Vector(m_inputSize));
      m_stats->min.fill(std::numeric_limits<double>::max());
      m_stats->min.fill(std::numeric_limits<double>::min());
    }

    for (size_t i = 0; ss.good(); ++i) {
      if (i > m_inputSize) {
        EXCEPTION("Input too large");
      }

      std::string token;
      std::getline(ss, token, ',');

      if (i == 0 && token.length() > 0) {
        label = token;
      }
      else {
        double value = std::stod(token);
        sample[i - 1] = value;

        if (value < m_stats->min[i - 1]) {
          m_stats->min[i - 1] = value;
        }
        if (value > m_stats->max[i - 1]) {
          m_stats->max[i - 1] = value;
        }
      }
    }

    samples.emplace_back(label, sample);
    ++numSamples;

    if (numSamples >= N) {
      break;
    }
  }

  return numSamples;
}
