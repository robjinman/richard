#pragma once

#include <filesystem>
#include <memory>

namespace richard {

class PlatformPaths {
  public:
    virtual std::filesystem::path get(const std::string& directory) const = 0;
    virtual std::filesystem::path get(const std::string& directory,
      const std::string& name) const = 0;

    virtual ~PlatformPaths() {};
};

using PlatformPathsPtr = std::unique_ptr<PlatformPaths>;

PlatformPathsPtr createPlatformPaths();

}
