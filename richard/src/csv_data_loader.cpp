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
CsvDataLoader::CsvDataLoader(const std::string& filePath, size_t inputSize,
  const std::vector<std::string>& labels)
  : m_inputSize(inputSize)
  , m_fin(filePath) {}

void CsvDataLoader::seekToBeginning() {
  m_fin.seekg(0);
}

size_t CsvDataLoader::loadSamples(std::vector<Sample>& samples, size_t N) {
  size_t numSamples = 0;
  std::string line;
  while (std::getline(m_fin, line)) {
    std::stringstream ss{line};
    std::string label = "_";
    Vector sample(m_inputSize);

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
