#pragma once

#include <fstream>
#include "data_loader.hpp"
#include "data_details.hpp"

class CsvDataLoader : public DataLoader {
  public:
    CsvDataLoader(const std::string& filePath, size_t inputSize,
      const NormalizationParams& normalization);

    void seekToBeginning() override;
    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;

    // Exposed for testing
    size_t loadSamples(std::istream& i, std::vector<Sample>& samples, size_t n);

  private:
    size_t m_inputSize;
    NormalizationParams m_normalization;
    std::ifstream m_fin;
};

