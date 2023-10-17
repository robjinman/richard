#include <sstream>
#include "csv_data_loader.hpp"
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
CsvDataLoader::CsvDataLoader(const std::string& filePath, size_t inputSize)
  : m_inputSize(inputSize)
  , m_fin(filePath) {}

void CsvDataLoader::seekToBeginning() {
  m_fin.seekg(0);
}

size_t CsvDataLoader::loadSamples(std::vector<Sample>& samples, size_t N) {
  return loadSamples(m_fin, samples, N);
}

size_t CsvDataLoader::loadSamples(std::istream& stream, std::vector<Sample>& samples, size_t N) {
  size_t numSamples = 0;
  std::string line;
  while (std::getline(stream, line)) {
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
        double value = std::stod(token);
        v[i - 1] = value;
      }
    }

    Array3 asArr3(std::move(v.storage()), v.size(), 1, 1);

    samples.emplace_back(label, asArr3);
    ++numSamples;

    if (numSamples >= N) {
      break;
    }
  }

  return numSamples;
}
