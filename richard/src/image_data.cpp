#include <filesystem>
#include <iostream> // TODO
#include "image_data.hpp"
#include "dataset.hpp"
#include "bitmap.hpp"

using namespace cpputils;

void loadImageData(Dataset& data, const std::string& directoryPath, const std::string& label) {
  const size_t N = 100; // TODO

  size_t i = 0;
  for (const auto& dirEntry : std::filesystem::directory_iterator{directoryPath}) {
    if (std::filesystem::is_regular_file(dirEntry)) {
      std::cout << dirEntry.path() << "\n";

      Bitmap image = loadBitmap(dirEntry.path().string());
      Vector v(image.numElements());

      for (size_t i = 0; i < image.numElements(); ++i) {
        v[i] = static_cast<double>(image.data[i]);
      }

      data.addSample(label, v);
    }

    if (++i > N) {
      break;
    }
  }
}
