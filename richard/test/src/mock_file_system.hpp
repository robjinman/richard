#pragma once

#include <file_system.hpp>
#include <gmock/gmock.h>

class MockFileSystem : public FileSystem {
  public:
    MOCK_METHOD(std::unique_ptr<std::ostream>, openFileForWriting, (const std::string&),
      (override));
    MOCK_METHOD(std::unique_ptr<std::istream>, openFileForReading, (const std::string&),
      (override));
};

