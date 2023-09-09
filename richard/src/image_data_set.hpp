#pragma once

#include <vector>
#include <filesystem>
#include "labelled_data_set.hpp"

class ImageDataSet : public LabelledDataSet {
  public:
    ImageDataSet(const std::string& directoryPath, const std::vector<std::string>& labels);

    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;
    void seekToBeginning() override;
    const DataStats& stats() const override;

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
    std::unique_ptr<DataStats> m_stats;
};
