#pragma once

#include <data_loader.hpp>
#include <gmock/gmock.h>

class MockDataLoader : public DataLoader {
  public:
    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

