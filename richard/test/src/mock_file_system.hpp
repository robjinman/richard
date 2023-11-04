#pragma once

#include <gmock/gmock.h>
#include <file_system.hpp>

class MockFileSystem : public FileSystem {
  public:
    MOCK_METHOD(std::unique_ptr<std::ostream>, openFileForWriting, (const std::string&),
      (override));
    MOCK_METHOD(std::unique_ptr<std::istream>, openFileForReading, (const std::string&),
      (override));
};

