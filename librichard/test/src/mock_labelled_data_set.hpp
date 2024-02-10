#pragma once

#include <richard/labelled_data_set.hpp>
#include <gmock/gmock.h>

using namespace richard;

class MockLabelledDataSet : public LabelledDataSet {
  public:
    MockLabelledDataSet(DataLoaderPtr dataLoader, const std::vector<std::string>& labels)
      : LabelledDataSet(std::move(dataLoader), labels) {}

    MOCK_METHOD(std::vector<Sample>, loadSamples, (), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

