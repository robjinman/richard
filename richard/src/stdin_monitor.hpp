#pragma once

#include <functional>
#include <thread>
#include <mutex>
#include <map>

class StdinMonitor {
  public:
    StdinMonitor();
    void onKey(char c, std::function<void()> handler);

  private:
    std::mutex m_mutex;
    std::map<char, std::function<void()>> m_handlers;

    void waitForInput();
};

