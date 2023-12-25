#pragma once

#include <fstream>
#include <memory>

namespace richard {

class FileSystem {
  public:
    virtual std::unique_ptr<std::ostream> openFileForWriting(const std::string& path) = 0;
    virtual std::unique_ptr<std::istream> openFileForReading(const std::string& path) = 0;

    virtual ~FileSystem() {}
};

using FileSystemPtr = std::unique_ptr<FileSystem>;

FileSystemPtr createFileSystem();

}
