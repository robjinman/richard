#pragma once

#include <memory>
#include <string>

namespace richard {

class Application {
  public:
    virtual std::string name() const = 0;
    virtual void start() = 0;

    virtual ~Application() {}
};

using ApplicationPtr = std::unique_ptr<Application>;

}
