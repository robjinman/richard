#include <fstream>
#include <cassert>
#include "bitmap.hpp"
#include "exception.hpp"

namespace cpputils {

Bitmap loadBitmap(const std::string& path) {
  BmpHeader bmpHeader(0, 0, 0, 0);

  size_t headerSize = sizeof(BmpHeader);

  std::ifstream fin(path, std::ios::binary);
  if (!fin.good()) {
    EXCEPTION("Error loading bitmap from " << path);
  }

  fin.read(reinterpret_cast<char*>(&bmpHeader), headerSize);

  uint32_t channels = bmpHeader.imgHdr.bitCount / 8;

  size_t size[3];
  size[0] = bmpHeader.imgHdr.height; // Rows
  size[1] = bmpHeader.imgHdr.width;  // Columns
  size[2] = channels;

  size_t bytes = size[0] * size[1] * size[2];
  uint8_t* data = new uint8_t[bytes];

  fin.seekg(bmpHeader.fileHdr.offset);

  size_t rowBytes = size[1] * channels;
  size_t paddedRowBytes = static_cast<size_t>(ceil(0.25 * rowBytes)) * 4;
  size_t rowPadding = paddedRowBytes - rowBytes;

  char* ptr = reinterpret_cast<char*>(data);
  for (size_t row = 0; row < size[0]; ++row) {
    fin.read(ptr, rowBytes);
    fin.ignore(rowPadding);
    ptr += rowBytes;
  }

  return Bitmap(data, size);
}

void saveBitmap(const Bitmap& bitmap, const std::string& path) {
  std::ofstream fout(path, std::ios::binary);
  if (!fout.good()) {
    EXCEPTION("Error saving bitmap at " << path);
  }

  uint32_t rows = static_cast<uint32_t>(bitmap.size()[0]);
  uint32_t cols = static_cast<uint32_t>(bitmap.size()[1]);
  uint16_t channels = static_cast<uint16_t>(bitmap.size()[2]);
  uint32_t paddedRowSize = static_cast<uint32_t>(ceil(0.25 * cols * channels)) * 4;
  uint32_t rowPadding = paddedRowSize - cols * channels;
  uint32_t rawSize = rows * paddedRowSize * channels;

  BmpHeader bmpHeader(cols, rows, channels, rawSize);
  fout.write(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));

  char zeros[4];
  memset(zeros, 0, 4);

  const uint8_t* ptr = bitmap.data;
  for (size_t row = 0; row < rows; ++row) {
    fout.write(reinterpret_cast<const char*>(ptr), cols * channels);
    ptr += cols * channels;

    assert(rowPadding <= 4);
    fout.write(zeros, rowPadding);
  }
}

}
