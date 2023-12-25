#include "file_system.hpp"

namespace richard {

class FileSystemImpl : public FileSystem {
  public:
    std::unique_ptr<std::ostream> openFileForWriting(const std::string& path) override;
    std::unique_ptr<std::istream> openFileForReading(const std::string& path) override;
};

std::unique_ptr<std::ostream> FileSystemImpl::openFileForWriting(const std::string& path) {
  return std::make_unique<std::ofstream>(path);
}

std::unique_ptr<std::istream> FileSystemImpl::openFileForReading(const std::string& path) {
  return std::make_unique<std::ifstream>(path);
}

FileSystemPtr createFileSystem() {
  return std::make_unique<FileSystemImpl>();
}

}
