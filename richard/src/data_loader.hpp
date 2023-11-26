#pragma once

#include "math.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include <memory>

struct Sample {
  Sample(const std::string& label, const Array3& data)
    : label(label)
    , data(data) {}

  std::string label; // TODO: Replace with reference/pointer/id
  Array3 data;
};

class DataLoader {
  public:
    virtual size_t loadSamples(std::vector<Sample>& samples) = 0;
    virtual void seekToBeginning() = 0;

    virtual ~DataLoader() {}
    
    static const nlohmann::json& exampleConfig();
};

using DataLoaderPtr = std::unique_ptr<DataLoader>;

class FileSystem;
class DataDetails;

DataLoaderPtr createDataLoader(FileSystem& fileSystem, const nlohmann::json& config,
  const std::string& samplesPath, const DataDetails& dataDetails);

