#include "labelled_data_set.hpp"
#include "exception.hpp"
#include "csv_data_loader.hpp"
#include "image_data_loader.hpp"
#include "file_system.hpp"

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

size_t LabelledDataSet::loadSamples(std::vector<Sample>& samples, size_t n) {
  samples.clear();
  return m_loader->loadSamples(samples, n);
}

std::unique_ptr<LabelledDataSet> createDataSet(FileSystem& fileSystem,
  const std::string& samplesPath, const DataDetails& dataDetails) {

  DataLoaderPtr dataLoader = nullptr;
  if (std::filesystem::is_directory(samplesPath)) {
    dataLoader = std::make_unique<ImageDataLoader>(samplesPath, dataDetails.classLabels,
      dataDetails.normalization);
  }
  else {
    const Triple& shape = dataDetails.shape;
    size_t inputSize = shape[0] * shape[1] * shape[2];
    
    auto fin = fileSystem.openFileForReading(samplesPath);

    dataLoader = std::make_unique<CsvDataLoader>(std::move(fin), inputSize,
      dataDetails.normalization);
  }

  return std::make_unique<LabelledDataSet>(std::move(dataLoader), dataDetails.classLabels);
}

