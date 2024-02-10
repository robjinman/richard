#include "richard/csv_data_loader.hpp"
#include "richard/exception.hpp"
#include <sstream>

namespace richard {

// Load training data from csv file
//
// Each line is a label followed by data values.
// E.g.
//
// b,23.1,45.5
// a,44.0,52.1
// c,11.9,92.4
// ...
CsvDataLoader::CsvDataLoader(std::unique_ptr<std::istream> stream, size_t inputSize,
  const NormalizationParams& normalization, size_t fetchSize)
  : DataLoader(fetchSize)
  , m_inputSize(inputSize)
  , m_normalization(normalization)
  , m_stream(std::move(stream)) {}

void CsvDataLoader::seekToBeginning() {
  m_stream->seekg(0);
}

std::vector<Sample> CsvDataLoader::loadSamples() {
  std::vector<Sample> samples;

  std::string line;
  while (std::getline(*m_stream, line)) {
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
        netfloat_t value = std::stof(token);
        v[i - 1] = normalize(m_normalization, value);
      }
    }

    Array3 asArr3(std::move(v.storage()), v.size(), 1, 1);

    samples.emplace_back(label, asArr3);

    if (samples.size() >= fetchSize()) {
      break;
    }
  }

  return samples;
}

}
