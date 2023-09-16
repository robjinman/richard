#pragma once

#include <filesystem>
#include "data_loader.hpp"

class ImageDataLoader : public DataLoader {
  public:
    ImageDataLoader(const std::string& directoryPath, const std::vector<std::string>& labels);

    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;
    void seekToBeginning() override;

  private:
    struct ClassCursor {
      ClassCursor(const std::string& label, const std::filesystem::directory_iterator& i)
        : label(label)
        , i(i) {}

      std::string label;
      std::filesystem::directory_iterator i;
    };

    std::filesystem::path m_directoryPath;
    std::vector<ClassCursor> m_iterators;
};
