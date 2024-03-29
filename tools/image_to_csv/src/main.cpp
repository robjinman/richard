#include <cpputils/bitmap.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace cpputils;

namespace {

const std::string DESCRIPTION = "Convert between csv and bmp";

void bmpToCsv(const std::filesystem::path& bmpFilePath) {
  Bitmap bm = loadBitmap(bmpFilePath);

  size_t Y = bm.size()[0];
  size_t X = bm.size()[1];
  for (size_t y = 0; y < Y; ++y) {
    for (size_t x = 0; x < X; ++x) {
      std::cout << static_cast<size_t>(bm[y][x][0]);
      bool lastValue = y + 1 == Y && x + 1 == X;
      if (!lastValue) {
        std::cout << ",";
      }
    }
  }
  std::cout << std::endl;
}

}

//csvbmp ./my_bitmaps
//csvbmp ./file.csv ./my_bitmaps

int main(int argc, char** argv) {
  bool csvToBmpMode = false;
  if (argc == 2) {
    csvToBmpMode = false;
  }
  else if (argc == 5) {
    csvToBmpMode = true;
  }
  else {
    std::cout << "Usage:" << std::endl
      << "\t" << argv[0] << " bitmaps_dir|bitmap_file" << std::endl
      << "\t" << argv[0] << " csv_file output_dir bmpWidth bmpHeight" << std::endl;
    return 1;
  }

  if (csvToBmpMode) {
    std::string csvFile = argv[1];
    std::filesystem::path outputDir(argv[2]);
    std::string strWidth = argv[3];
    std::string strHeight = argv[4];
    size_t W = std::stoul(strWidth);
    size_t H = std::stoul(strHeight);
    size_t bmSize[] = { H, W, 3 };

    std::ifstream stream(csvFile);

    std::string line;
    std::getline(stream, line);

    std::stringstream ssClasses{line};
    while (ssClasses.good()) {
      std::string token;
      std::getline(ssClasses, token, ',');

      std::filesystem::create_directories(outputDir/token);
    }

    size_t imageId = 0;
    while (std::getline(stream, line)) {
      std::stringstream ssData{line};
      std::string label = "0";
      std::getline(ssData, label, ',');

      Bitmap bm(bmSize);

      for (size_t i = 0; ssData.good(); ++i) {
        std::string token;
        std::getline(ssData, token, ',');

        size_t row = (H - 1 - i / W);
        size_t col = i % W;

        bm[row][col][0] = static_cast<uint8_t>(std::stoi(token));
        bm[row][col][1] = static_cast<uint8_t>(std::stoi(token));
        bm[row][col][2] = static_cast<uint8_t>(std::stoi(token));
      }

      std::filesystem::path outputPath = outputDir/label/(std::to_string(imageId) + ".bmp");
      saveBitmap(bm, outputPath);

      ++imageId;
    }
  }
  else {
    std::string inputFilePath = argv[1];

    if (std::filesystem::is_directory(inputFilePath)) {
      for (auto const& dirEntry : std::filesystem::directory_iterator{inputFilePath}) {
        if (std::filesystem::is_regular_file(dirEntry)) {
          bmpToCsv(dirEntry.path());
        }
      }
    }
    else {
      bmpToCsv(inputFilePath);
    }
  }

  return 0;
}
