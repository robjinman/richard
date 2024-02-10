#pragma once

#include "richard/math.hpp"
#include "richard/data_loader.hpp"
#include "richard/types.hpp"
#include <map>
#include <string>
#include <memory>

namespace richard {

class DataDetails;
class FileSystem;

class LabelledDataSet {
  public:
    LabelledDataSet(DataLoaderPtr loader, const std::vector<std::string>& labels);

    virtual std::vector<Sample> loadSamples();
    virtual void seekToBeginning();

    inline const std::vector<std::string>& labels() const;
    inline const Vector& classOutputVector(const std::string& label) const;
    inline size_t fetchSize() const;

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

inline size_t LabelledDataSet::fetchSize() const {
  return m_loader->fetchSize();
}

}
