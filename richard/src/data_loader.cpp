#include "data_loader.hpp"
#include "utils.hpp"
#include "image_data_loader.hpp"
#include "csv_data_loader.hpp"
#include "file_system.hpp"

namespace richard {

DataLoader::DataLoader(size_t fetchSize)
  : m_fetchSize(fetchSize) {}

const Config& DataLoader::exampleConfig() {
  static Config obj;
  static bool done = false;
  
  if (!done) {
    obj.setInteger("fetchSize", 500);

    done = true;
  }
  
  return obj;
}

DataLoaderPtr createDataLoader(FileSystem& fileSystem, const Config& config,
  const std::string& samplesPath, const DataDetails& dataDetails) {

  size_t fetchSize = config.getInteger("fetchSize");

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
