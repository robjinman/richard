#pragma once

#include <vector>
#include "math.hpp"

struct Sample {
  Sample(const std::string& label, const Array3& data)
    : label(label)
    , data(data) {}

  std::string label; // TODO: Replace with reference/pointer/id
  Array3 data;
};

class DataLoader {
  public:
    virtual size_t loadSamples(std::vector<Sample>& samples, size_t n) = 0;
    virtual void seekToBeginning() = 0;

    virtual ~DataLoader() {}
};
