#include "file_system.hpp"
#include <sstream>

namespace fs = std::filesystem;

namespace richard {

class FileSystemImpl : public FileSystem {
  public:
    std::unique_ptr<std::ostream> openFileForWriting(const fs::path& path) override;
    std::unique_ptr<std::istream> openFileForReading(const fs::path& path) override;

    std::string loadTextFile(const std::filesystem::path& path) override;
};

std::unique_ptr<std::ostream> FileSystemImpl::openFileForWriting(const fs::path& path) {
  return std::make_unique<std::ofstream>(path, std::ios::binary);
}

std::unique_ptr<std::istream> FileSystemImpl::openFileForReading(const fs::path& path) {
  return std::make_unique<std::ifstream>(path, std::ios::binary);
}

std::string FileSystemImpl::loadTextFile(const fs::path& path) {
  std::ifstream stream(path);
  std::stringstream ss;
  std::string line;
  while (std::getline(stream, line)) {
    ss << line << std::endl;
  }
  return ss.str();
}

FileSystemPtr createFileSystem() {
  return std::make_unique<FileSystemImpl>();
}

}
