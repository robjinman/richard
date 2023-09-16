#pragma once

#include <map>
#include <string>
#include "math.hpp"
#include "data_loader.hpp"

class LabelledDataSet {
  public:
    LabelledDataSet(const std::vector<std::string>& labels);

    virtual size_t loadSamples(std::vector<Sample>& samples, size_t n) = 0;
    virtual void seekToBeginning() = 0;

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
