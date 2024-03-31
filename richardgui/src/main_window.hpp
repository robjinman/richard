#pragma once

#include <wx/wx.h>

class MainWindow : public wxFrame {
  public:
    MainWindow();
 
  private:
    void OnHello(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
};
