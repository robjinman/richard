#include <iostream>
#include "stdin_monitor.hpp"

void StdinMonitor::onKey(char c, std::function<void()> handler) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_handlers[c] = handler;
}

StdinMonitor::StdinMonitor() {
  std::thread t(&StdinMonitor::waitForInput, this);
  t.detach();
}

void StdinMonitor::waitForInput() {
  while (true) {
    char c = '\0';
    std::cin >> c;

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto i = m_handlers.find(c);
      if (i != m_handlers.end()) {
        i->second();
      }
    }
  }
}

