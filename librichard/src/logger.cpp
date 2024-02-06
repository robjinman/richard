#include "richard/logger.hpp"
#include <iostream>

namespace richard {

class StdoutLogger : public Logger {
  public:
    void debug(const std::string& msg, bool newline = true) override;
    void info(const std::string& msg, bool newline = true) override;
    void warn(const std::string& msg, bool newline = true) override;
    void error(const std::string& msg, bool newline = true) override;
    
  private:
    void endMessage(bool newline) const;
};

void StdoutLogger::endMessage(bool newline) const {
  if (newline) {
    std::cout << std::endl;
  }
  else {
    std::cout << std::flush;
  }
}

void StdoutLogger::debug(const std::string& msg, bool newline) {
  std::cout << "[ DEBUG ] " << msg;
  endMessage(newline);
}

void StdoutLogger::info(const std::string& msg, bool newline) {
  std::cout << msg;
  endMessage(newline);
}

void StdoutLogger::warn(const std::string& msg, bool newline) {
  std::cerr << "Warning: " << msg;
  endMessage(newline);
}

void StdoutLogger::error(const std::string& msg, bool newline) {
  std::cerr << "Error: " << msg;
  endMessage(newline);
}

LoggerPtr createStdoutLogger() {
  return std::make_unique<StdoutLogger>();
}

}
