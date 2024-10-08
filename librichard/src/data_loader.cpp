#include "richard/data_loader.hpp"
#include "richard/utils.hpp"
#include "richard/image_data_loader.hpp"
#include "richard/csv_data_loader.hpp"
#include "richard/file_system.hpp"

namespace richard {

DataLoader::DataLoader(size_t fetchSize)
  : m_fetchSize(fetchSize) {}

const Config& DataLoader::exampleConfig() {
  static Config config = []() {
    Config c;
    c.setNumber("fetchSize", 500);
    return c;
  }();
  
  return config;
}

DataLoaderPtr createDataLoader(FileSystem& fileSystem, const Config& config,
  const std::string& samplesPath, const DataDetails& dataDetails) {

  size_t fetchSize = config.getNumber<size_t>("fetchSize");

  if (std::filesystem::is_directory(samplesPath)) {
    return std::make_unique<ImageDataLoader>(samplesPath, dataDetails.classLabels,
      dataDetails.normalization, fetchSize);
  }
  else {
    auto stream = fileSystem.openFileForReading(samplesPath);

    return std::make_unique<CsvDataLoader>(std::move(stream), calcProduct(dataDetails.shape),
      dataDetails.normalization, fetchSize);
  }
}

}
