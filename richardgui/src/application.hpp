#pragma once

#include "main_window.hpp"
#include <wx/wx.h>

class Application : public wxApp {
  public:
    bool OnInit() override;
};
