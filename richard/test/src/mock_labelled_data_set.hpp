#pragma once

#include <gmock/gmock.h>
#include <labelled_data_set.hpp>

class MockLabelledDataSet : public LabelledDataSet {
  public:
    MockLabelledDataSet(DataLoaderPtr dataLoader, const std::vector<std::string>& labels)
      : LabelledDataSet(std::move(dataLoader), labels) {}

    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

