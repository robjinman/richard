#pragma once

#include <richard/data_loader.hpp>
#include <gmock/gmock.h>

using namespace richard;

class MockDataLoader : public DataLoader {
  public:
    MockDataLoader()
      : DataLoader(128) {}

    MOCK_METHOD(std::vector<Sample>, loadSamples, (), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

