#pragma once

#include <data_loader.hpp>
#include <gmock/gmock.h>

using namespace richard;

class MockDataLoader : public DataLoader {
  public:
    MockDataLoader()
      : DataLoader(128) {}

    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

