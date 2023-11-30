#pragma once

#include <string>
#include <memory>

class Logger {
  public:
    virtual void info(const std::string& msg, bool newline = true) = 0;
    virtual void warn(const std::string& msg, bool newline = true) = 0;
    virtual void error(const std::string& msg, bool newline = true) = 0;

    virtual ~Logger() {}
};

using LoggerPtr = std::unique_ptr<Logger>;

LoggerPtr createStdoutLogger();

