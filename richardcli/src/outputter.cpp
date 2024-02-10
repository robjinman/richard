#include "outputter.hpp"
#include <richard/utils.hpp>

using namespace richard;

Outputter::Outputter(std::ostream& stream)
  : m_stream(stream) {}

void Outputter::printBanner() {
  m_stream
    << R"( ___ _    _                _ )" << std::endl 
    << R"(| _ (_)__| |_  __ _ _ _ __| |)" << std::endl
    << R"(|   / / _| ' \/ _` | '_/ _` |)" << std::endl
    << R"(|_|_\_\__|_||_\__,_|_| \__,_|)" << std::endl
    <<  "v" << versionString() << std::endl;
}

void Outputter::printSeparator() {
  m_stream << std::string(80, '-') << std::endl;
}

void Outputter::printLine(const std::string& line, bool newline) {
  m_stream << line;
  if (newline) {
    m_stream << std::endl;
  }
  else {
    m_stream << std::flush;
  }
}
