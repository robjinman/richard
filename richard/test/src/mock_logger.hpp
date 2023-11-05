#pragma once

#include <gmock/gmock.h>
#include <logger.hpp>

class MockLogger : public Logger {
  public:
    MOCK_METHOD(void, info, (const std::string&, bool), (override));
    MOCK_METHOD(void, warn, (const std::string&, bool), (override));
    MOCK_METHOD(void, error, (const std::string&, bool), (override));
};

