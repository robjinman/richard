#pragma once

#include <string>
#include <vector>
#include <memory>
#include "dataset.hpp"

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
  const std::vector<std::string>& classLabels);
