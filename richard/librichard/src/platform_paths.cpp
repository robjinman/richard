#include "platform_paths.hpp"
#include "exception.hpp"
#include <map>

namespace fs = std::filesystem;

namespace richard {
namespace {

const fs::path& assertExists(const fs::path& path) {
  ASSERT_MSG(fs::exists(path), "Path " << path << " does not exist");
  return path;
}

}

class LinuxPaths : public PlatformPaths {
  public:
    LinuxPaths();

    fs::path get(const std::string& directory) const override;
    fs::path get(const std::string& directory, const std::string& name) const override;

  private:
    std::map<std::string, fs::path> m_directories;
};

LinuxPaths::LinuxPaths() {
  m_directories["shaders"] = assertExists(fs::current_path().append("shaders"));
}

fs::path LinuxPaths::get(const std::string& directory) const {
  auto i = m_directories.find(directory);
  ASSERT_MSG(i != m_directories.end(), "Unrecognised application directory: " << directory);

  return i->second;
}

fs::path LinuxPaths::get(const std::string& directory, const std::string& name) const {
  auto dir = get(directory);
  return assertExists(dir.append(name));
}

PlatformPathsPtr createPlatformPaths() {
#ifdef WIN32
  // There's currently no difference between Windows and Linux
  return std::make_unique<LinuxPaths>();
#else
  return std::make_unique<LinuxPaths>();
#endif
}

}
