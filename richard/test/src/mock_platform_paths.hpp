#pragma once

#include <platform_paths.hpp>
#include <gmock/gmock.h>

using namespace richard;

class MockPlatformPaths : public PlatformPaths {
  public:
    MOCK_METHOD(std::filesystem::path, get, (const std::string& directory), (const, override));
    MOCK_METHOD(std::filesystem::path, get, (const std::string& directory, const std::string& name),
      (const, override));
};
