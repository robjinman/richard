#pragma once

#include <map>
#include <string>
#include <vector>
#include "math.hpp"

struct Sample {
  Sample(const std::string& label, const Vector& data)
    : label(label)
    , data(data) {}

  std::string label;
  Vector data;
};

class Dataset {
  public:
    explicit Dataset(const std::vector<std::string>& labels);

    inline void addSample(const std::string& label, const Vector& data);
    inline const Vector& classOutputVector(const std::string& label) const;
    inline std::vector<Sample>& samples();
    inline const std::vector<Sample>& samples() const;
    void normalize(const Vector& min, const Vector& max);

  private:
    std::vector<std::string> m_labels;
    std::map<std::string, Vector> m_classOutputVectors;
    std::vector<Sample> m_samples;
};

inline void Dataset::addSample(const std::string& label, const Vector& data) {
  ASSERT(m_classOutputVectors.count(label));
  m_samples.emplace_back(label, data);
}

inline const std::vector<Sample>& Dataset::samples() const {
  return m_samples;
}

inline const Vector& Dataset::classOutputVector(const std::string& label) const {
  return m_classOutputVectors.at(label);
}

inline std::vector<Sample>& Dataset::samples() {
  return m_samples;
}

class TrainingData {
  public:
    explicit TrainingData(std::unique_ptr<Dataset> data);

    inline const Dataset& data() const;
    void normalize();
    inline const Vector& min() const;
    inline const Vector& max() const;

  private:
    std::unique_ptr<Dataset> m_data;
    Vector m_min;
    Vector m_max;
};

inline const Dataset& TrainingData::data() const {
  return *m_data;
}

inline const Vector& TrainingData::min() const {
  return m_min;
}

inline const Vector& TrainingData::max() const {
  return m_max;
}

class TestData {
  public:
    explicit TestData(std::unique_ptr<Dataset> data);

    inline const Dataset& data() const;
    void normalize(const Vector& trainingDataMin, const Vector& trainingDataMax);

  private:
    std::unique_ptr<Dataset> m_data;
};

inline const Dataset& TestData::data() const {
  return *m_data;
}
