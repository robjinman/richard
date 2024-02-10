#include "richard/logger.hpp"
#include <iostream>

namespace richard {

class LoggerImpl : public Logger {
  public:
    LoggerImpl(std::ostream& errorStream, std::ostream& warningStream, std::ostream& infoStream,
      std::ostream& debugStream)
      : m_error(errorStream)
      , m_warning(warningStream)
      , m_info(infoStream)
      , m_debug(debugStream) {}

    void debug(const std::string& msg, bool newline = true) override;
    void info(const std::string& msg, bool newline = true) override;
    void warn(const std::string& msg, bool newline = true) override;
    void error(const std::string& msg, bool newline = true) override;
    
  private:
    std::ostream& m_error;
    std::ostream& m_warning;
    std::ostream& m_info;
    std::ostream& m_debug;

    void endMessage(std::ostream& stream, bool newline) const;
};

void LoggerImpl::endMessage(std::ostream& stream, bool newline) const {
  if (newline) {
    stream << std::endl;
  }
  else {
    stream << std::flush;
  }
}

void LoggerImpl::debug(const std::string& msg, bool newline) {
  m_debug << "[ DEBUG ] " << msg;
  endMessage(m_debug, newline);
}

void LoggerImpl::info(const std::string& msg, bool newline) {
  m_info << "[ INFO ] " << msg;
  endMessage(m_info, newline);
}

void LoggerImpl::warn(const std::string& msg, bool newline) {
  m_warning << "[ WARNING ] " << msg;
  endMessage(m_warning, newline);
}

void LoggerImpl::error(const std::string& msg, bool newline) {
  m_error << "[ ERROR ] " << msg;
  endMessage(m_error, newline);
}

LoggerPtr createLogger(std::ostream& errorStream, std::ostream& warningStream,
  std::ostream& infoStream, std::ostream& debugStream) {

  return std::make_unique<LoggerImpl>(errorStream, warningStream, infoStream, debugStream);
}

}
