#pragma once

#include "data_loader.hpp"
#include "data_details.hpp"
#include <filesystem>

namespace richard {

class ImageDataLoader : public DataLoader {
  public:
    ImageDataLoader(const std::string& directoryPath, const std::vector<std::string>& labels,
      const NormalizationParams& normalization, size_t fetchSize);

    size_t loadSamples(std::vector<Sample>& samples) override;
    void seekToBeginning() override;

  private:
    struct ClassCursor {
      ClassCursor(const std::string& label, const std::filesystem::directory_iterator& i)
        : label(label)
        , i(i) {}

      std::string label;
      std::filesystem::directory_iterator i;
    };

    NormalizationParams m_normalization;
    std::filesystem::path m_directoryPath;
    std::vector<ClassCursor> m_iterators;
};

}
