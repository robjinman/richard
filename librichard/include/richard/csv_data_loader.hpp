#pragma once

#include "richard/data_loader.hpp"
#include "richard/data_details.hpp"
#include <fstream>
#include <memory>

namespace richard {

class CsvDataLoader : public DataLoader {
  public:
    CsvDataLoader(std::unique_ptr<std::istream>, size_t inputSize,
      const NormalizationParams& normalization, size_t fetchSize);

    void seekToBeginning() override;
    std::vector<Sample> loadSamples() override;

  private:
    size_t m_inputSize;
    NormalizationParams m_normalization;
    std::unique_ptr<std::istream> m_stream;
};

}
