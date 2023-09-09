#pragma once

#include <map>
#include <string>
#include <vector>
#include "math.hpp"

struct Sample {
  Sample(const std::string& label, const Vector& data)
    : label(label)
    , data(data) {}

  std::string label; // TODO: Replace with reference/pointer/id
  Vector data;
};

struct DataStats {
  DataStats(const Vector& min, const Vector& max)
    : min(min)
    , max(max) {}

  Vector min; // Min/max values of every dimension
  Vector max;
};

class LabelledDataSet {
  public:
    LabelledDataSet(const std::vector<std::string>& labels);

    virtual size_t loadSamples(std::vector<Sample>& samples, size_t n) = 0;
    virtual void seekToBeginning() = 0;
    virtual const DataStats& stats() const = 0;

    inline const std::vector<std::string>& labels() const;
    inline const Vector& classOutputVector(const std::string& label) const;

    virtual ~LabelledDataSet() = 0;

  private:
    std::vector<std::string> m_labels;
    std::map<std::string, Vector> m_classOutputVectors;
};

inline const std::vector<std::string>& LabelledDataSet::labels() const {
  return m_labels;
}

inline const Vector& LabelledDataSet::classOutputVector(const std::string& label) const {
  return m_classOutputVectors.at(label);
}

void normalizeTrainingSamples(std::vector<Sample>& samples);
void normalizeTestSamples(std::vector<Sample>& samples, const Vector& min, const Vector& max);
