#include "richard/file_system.hpp"
#include "richard/exception.hpp"
#include <sstream>

namespace fs = std::filesystem;

namespace richard {

class FileSystemImpl : public FileSystem {
  public:
    std::unique_ptr<std::ostream> openFileForWriting(const fs::path& path) override;
    std::unique_ptr<std::istream> openFileForReading(const fs::path& path) override;

    std::string loadTextFile(const std::filesystem::path& path) override;
    std::vector<uint8_t> loadBinaryFile(const std::filesystem::path& path) override;
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

std::vector<uint8_t> FileSystemImpl::loadBinaryFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!stream.read(reinterpret_cast<char*>(buffer.data()), size)) {
    EXCEPTION("Failed to load file at '" << path << "'");
  }

  return buffer;
}

FileSystemPtr createFileSystem() {
  return std::make_unique<FileSystemImpl>();
}

}
