#pragma once

#include <richard/file_system.hpp>
#include <gmock/gmock.h>

using namespace richard;

class MockFileSystem : public FileSystem {
  public:
    MOCK_METHOD(std::unique_ptr<std::ostream>, openFileForWriting, (const std::filesystem::path&),
      (override));
    MOCK_METHOD(std::unique_ptr<std::istream>, openFileForReading, (const std::filesystem::path&),
      (override));
    MOCK_METHOD(std::string, loadTextFile, (const std::filesystem::path&), (override));
    MOCK_METHOD(std::vector<uint8_t>, loadBinaryFile, (const std::filesystem::path&), (override));
};
