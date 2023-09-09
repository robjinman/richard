#pragma once

#include <fstream>
#include "labelled_data_set.hpp"

class CsvDataSet : public LabelledDataSet {
  public:
    CsvDataSet(const std::string& filePath, size_t inputSize,
      const std::vector<std::string>& labels);

    void seekToBeginning() override;
    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;
    const DataStats& stats() const override;

  private:
    size_t m_inputSize;
    std::ifstream m_fin;
    std::unique_ptr<DataStats> m_stats;
};
