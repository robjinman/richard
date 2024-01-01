#include "data_loader.hpp"
#include "utils.hpp"
#include "image_data_loader.hpp"
#include "csv_data_loader.hpp"
#include "file_system.hpp"

namespace richard {

DataLoader::DataLoader(size_t fetchSize)
  : m_fetchSize(fetchSize) {}

const nlohmann::json& DataLoader::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;
  
  if (!done) {
    obj["fetchSize"] = 500;

    done = true;
  }
  
  return obj;
}

DataLoaderPtr createDataLoader(FileSystem& fileSystem, const nlohmann::json& config,
  const std::string& samplesPath, const DataDetails& dataDetails) {

  size_t fetchSize = getOrThrow(config, "fetchSize").get<size_t>();

  if (std::filesystem::is_directory(samplesPath)) {
    return std::make_unique<ImageDataLoader>(samplesPath, dataDetails.classLabels,
      dataDetails.normalization, fetchSize);
  }
  else {
    const Size3& shape = dataDetails.shape;
    size_t inputSize = shape[0] * shape[1] * shape[2];

    auto fin = fileSystem.openFileForReading(samplesPath);

    return std::make_unique<CsvDataLoader>(std::move(fin), inputSize, dataDetails.normalization,
      fetchSize);
  }
}

}
