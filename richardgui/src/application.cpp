#include "application.hpp"

wxIMPLEMENT_APP(Application);

bool Application::OnInit() {
  // Framework takes ownership of all windows. Do not delete.
  MainWindow* window = new MainWindow;
  window->Show(true);

  return true;
}
