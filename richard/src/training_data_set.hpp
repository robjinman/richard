#pragma once

#include "labelled_data_set.hpp"
#include "data_stats.hpp"

class TrainingDataSet : public LabelledDataSet {
  public:
    TrainingDataSet(std::unique_ptr<DataLoader> loader, const std::vector<std::string>& labels,
      bool computeStatsAndNormalize);

    void seekToBeginning() override;
    size_t loadSamples(std::vector<Sample>& samples, size_t n) override;
    const DataStats& stats() const;

  private:
    std::unique_ptr<DataLoader> m_loader;
    bool m_computeStatsAndNormalize;
    std::unique_ptr<DataStats> m_stats;
};
