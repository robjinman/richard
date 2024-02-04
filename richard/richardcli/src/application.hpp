#pragma once

#include <memory>

namespace richard {

class Application {
  public:
    virtual void start() = 0;

    virtual ~Application() {}
};

using ApplicationPtr = std::unique_ptr<Application>;

}
