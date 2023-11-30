#include "csv_data_loader.hpp"
#include "exception.hpp"
#include <sstream>

// Load training data from csv file
//
// Each line is a label followed by data values.
// E.g.
//
// b,23.1,45.5
// a,44.0,52.1
// c,11.9,92.4
// ...
CsvDataLoader::CsvDataLoader(std::unique_ptr<std::istream> fin, size_t inputSize,
  const NormalizationParams& normalization, size_t fetchSize)
  : m_inputSize(inputSize)
  , m_normalization(normalization)
  , m_fetchSize(fetchSize)
  , m_fin(std::move(fin)) {}

void CsvDataLoader::seekToBeginning() {
  m_fin->seekg(0);
}

size_t CsvDataLoader::loadSamples(std::vector<Sample>& samples) {
  size_t numSamples = 0;
  std::string line;
  while (std::getline(*m_fin, line)) {
    std::stringstream ss{line};
    std::string label = "_";
    Vector v(m_inputSize);

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
        netfloat_t value = std::stod(token);
        v[i - 1] = normalize(m_normalization, value);
      }
    }

    Array3 asArr3(std::move(v.storage()), v.size(), 1, 1);

    samples.emplace_back(label, asArr3);
    ++numSamples;

    if (numSamples >= m_fetchSize) {
      break;
    }
  }

  return numSamples;
}

