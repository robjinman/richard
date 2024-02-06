#include "richard/labelled_data_set.hpp"
#include "richard/exception.hpp"
#include "richard/csv_data_loader.hpp"
#include "richard/image_data_loader.hpp"
#include "richard/file_system.hpp"

namespace richard {

LabelledDataSet::LabelledDataSet(DataLoaderPtr loader, const std::vector<std::string>& labels)
  : m_loader(std::move(loader))
  , m_labels(labels) {

  for (size_t i = 0; i < m_labels.size(); ++i) {
    Vector v(m_labels.size());
    v.zero();
    v[i] = 1.0;
    m_classOutputVectors.insert({m_labels[i], v});
  }
}

void LabelledDataSet::seekToBeginning() {
  m_loader->seekToBeginning();
}

size_t LabelledDataSet::loadSamples(std::vector<Sample>& samples) {
  samples.clear();
  return m_loader->loadSamples(samples);
}

}
