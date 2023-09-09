#include <fstream>
#include <sstream>
#include "csv_data.hpp"
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
std::unique_ptr<Dataset> loadCsvData(const std::string& filePath, size_t inputSize,
  const std::vector<std::string>& classLabels) {

  std::ifstream fin(filePath);

  auto data = std::make_unique<Dataset>(classLabels);

  std::string line;
  while (std::getline(fin, line)) {
    std::stringstream ss{line};
    std::string label = "_";
    Vector sample(inputSize);

    for (size_t i = 0; ss.good(); ++i) {
      if (i > inputSize) {
        EXCEPTION("Input too large");
      }

      std::string token;
      std::getline(ss, token, ',');

      if (i == 0 && token.length() > 0) {
        label = token;
      }
      else {
        sample[i - 1] = std::stod(token);
      }
    }

    data->addSample(label, sample);
  }

  return data;
}
