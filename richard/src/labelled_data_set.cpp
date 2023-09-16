#include "labelled_data_set.hpp"
#include "exception.hpp"

LabelledDataSet::LabelledDataSet(const std::vector<std::string>& labels)
  : m_labels(labels) {

  for (size_t i = 0; i < m_labels.size(); ++i) {
    Vector v(m_labels.size());
    v.zero();
    v[i] = 1.0;
    m_classOutputVectors.insert({m_labels[i], v});
  }
}

LabelledDataSet::~LabelledDataSet() {}
