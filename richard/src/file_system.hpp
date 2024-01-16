#pragma once

#include <fstream>
#include <memory>
#include <filesystem>

namespace richard {

class FileSystem {
  public:
    virtual std::unique_ptr<std::ostream> openFileForWriting(const std::filesystem::path& path) = 0;
    virtual std::unique_ptr<std::istream> openFileForReading(const std::filesystem::path& path) = 0;

    virtual std::string loadTextFile(const std::filesystem::path& path) = 0;

    virtual ~FileSystem() {}
};

using FileSystemPtr = std::unique_ptr<FileSystem>;

FileSystemPtr createFileSystem();

}
