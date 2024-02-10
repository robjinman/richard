#pragma once

#include <ostream>

class Outputter {
  public:
    Outputter(std::ostream& stream);

    void printBanner();
    void printSeparator();
    void printLine(const std::string& line, bool newline = true);

  private:
    std::ostream& m_stream;
};
