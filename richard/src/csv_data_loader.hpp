#pragma once

#include <fstream>
#include <memory>
#include "data_loader.hpp"
#include "data_details.hpp"

class CsvDataLoader : public DataLoader {
  public:
    CsvDataLoader(std::unique_ptr<std::istream>, size_t inputSize,
      const NormalizationParams& normalization);

    void seekToBeginning() override;
    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;

  private:
    size_t m_inputSize;
    NormalizationParams m_normalization;
    std::unique_ptr<std::istream> m_fin;
};

