#pragma once

#include <map>
#include <string>
#include <memory>
#include "math.hpp"
#include "data_loader.hpp"
#include "types.hpp"

class DataDetails;
class FileSystem;

class LabelledDataSet {
  public:
    LabelledDataSet(DataLoaderPtr loader, const std::vector<std::string>& labels);

    virtual size_t loadSamples(std::vector<Sample>& samples, size_t n);
    virtual void seekToBeginning();

    inline const std::vector<std::string>& labels() const;
    inline const Vector& classOutputVector(const std::string& label) const;

    virtual ~LabelledDataSet() {}

  private:
    DataLoaderPtr m_loader;
    std::vector<std::string> m_labels;
    std::map<std::string, Vector> m_classOutputVectors;
};

inline const std::vector<std::string>& LabelledDataSet::labels() const {
  return m_labels;
}

inline const Vector& LabelledDataSet::classOutputVector(const std::string& label) const {
  return m_classOutputVectors.at(label);
}

std::unique_ptr<LabelledDataSet> createDataSet(FileSystem& fileSystem,
  const std::string& samplesPath, const DataDetails& dataDetails);

