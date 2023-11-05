#pragma once

#include <gmock/gmock.h>
#include <data_loader.hpp>

class MockDataLoader : public DataLoader {
  public:
    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

