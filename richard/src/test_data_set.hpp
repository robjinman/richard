#pragma once

#include "labelled_data_set.hpp"
#include "data_stats.hpp"

class TestDataSet : public LabelledDataSet {
  public:
    TestDataSet(std::unique_ptr<DataLoader> loader, const std::vector<std::string>& labels);

    void normalize(const DataStats& trainingDataStats);

    void seekToBeginning() override;
    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;

  private:
    std::unique_ptr<DataLoader> m_loader;
    bool m_normalize;
    DataStats m_trainingDataStats;
};
