#pragma once

#include "cpputils/array.hpp"
#include <cstdint>
#include <cmath>
#include <filesystem>

namespace cpputils {

using Bitmap = ContigMultiArray<uint8_t, 3>;

const uint32_t BMP_HEADER_SIZE = 54;

#pragma pack(push, 1)
struct BmpFileHeader {
  BmpFileHeader(uint32_t w, uint32_t h, uint32_t channels)
    : size(BMP_HEADER_SIZE + h * static_cast<uint32_t>(ceil(0.25 * w * channels)) * 4) {}

  char type[2] = {'B', 'M'};
  uint32_t size;
  uint16_t reserved1 = 0;
  uint16_t reserved2 = 0;
  uint32_t offset = BMP_HEADER_SIZE;
};

struct BmpImgHeader {
  BmpImgHeader(uint32_t w, uint32_t h, uint16_t channels, uint32_t rawSize)
    : width(w),
      height(h),
      bitCount(channels * 8),
      imgSize(rawSize) {}

  uint32_t size = 40;
  uint32_t width;
  uint32_t height;
  uint16_t planes = 1;
  uint16_t bitCount;
  uint32_t compression = 0;
  uint32_t imgSize;
  uint32_t xPxPerMetre = 0;
  uint32_t yPxPerMetre = 0;
  uint32_t colMapEntriesUsed = 0;
  uint32_t numImportantColours = 0;
};

struct BmpHeader {
  BmpHeader(uint32_t imgW, uint32_t imgH, uint16_t channels, uint32_t rawSize)
    : fileHdr(imgW, imgH, channels),
      imgHdr(imgW, imgH, channels, rawSize) {}

  BmpFileHeader fileHdr;
  BmpImgHeader imgHdr;
};
#pragma pack(pop)

Bitmap loadBitmap(const std::filesystem::path& path);
void saveBitmap(const Bitmap& bitmap, const std::filesystem::path& path);

}
