#pragma once

#include <logger.hpp>
#include <gmock/gmock.h>

using namespace richard;

class MockLogger : public Logger {
  public:
    MOCK_METHOD(void, info, (const std::string&, bool), (override));
    MOCK_METHOD(void, warn, (const std::string&, bool), (override));
    MOCK_METHOD(void, error, (const std::string&, bool), (override));
};

