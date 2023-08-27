#include <cstdio>
#include <cstdlib>
#include <jpeglib.h>
#include <iostream>
#include <filesystem>
#include "bitmap.hpp"

using namespace cpputils;

namespace {

Bitmap readJpegFile(const std::string& filename) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPARRAY buffer;
  int rowStride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  FILE* infile = fopen(filename.c_str(), "rb");
  if (infile == nullptr) {
    std::cerr << "Error opening file: " << filename << std::endl;
    exit(1);
  }
  jpeg_stdio_src(&cinfo, infile);

  jpeg_read_header(&cinfo, TRUE);

  jpeg_start_decompress(&cinfo);
  rowStride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, rowStride, 1);

  size_t size[] = { cinfo.output_height, cinfo.output_width, cinfo.output_components };
  Bitmap bm(size);

  size_t j = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);

    for (int i = 0; i < rowStride; i += 3) {
      bm[cinfo.output_height - 1 - j][i / 3][2] = buffer[0][i];
      bm[cinfo.output_height - 1 - j][i / 3][1] = buffer[0][i + 1];
      bm[cinfo.output_height - 1 - j][i / 3][0] = buffer[0][i + 2];
    }
    ++j;
  }

  jpeg_finish_decompress(&cinfo);

  jpeg_destroy_decompress(&cinfo);
  fclose(infile);

  return bm;
}

Bitmap processImage(const Bitmap& srcImage) {
  const size_t W = 100;
  const size_t H = 100;

  size_t size[] = { W, H, 3 };
  Bitmap finalImage(size);

  for (size_t j = 0; j < H; ++j) {
    for (size_t i = 0; i < W; ++i) {
      double y = static_cast<double>(j) / H * srcImage.size()[0];
      double x = static_cast<double>(i) / W * srcImage.size()[1];

      size_t j_ = static_cast<size_t>(y);
      size_t i_ = static_cast<size_t>(x);

      finalImage[j][i][0] = srcImage[j_][i_][0];
      finalImage[j][i][1] = srcImage[j_][i_][1];
      finalImage[j][i][2] = srcImage[j_][i_][2];
    }
  }

  return finalImage;
}

}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " input_dir output_dir" << std::endl;
    return 1;
  }

  std::string inputDir(argv[1]);
  std::filesystem::path outputDir(argv[2]);

  std::filesystem::create_directories(outputDir);

  for (const auto& dirEntry : std::filesystem::directory_iterator{inputDir}) {
    if (std::filesystem::is_regular_file(dirEntry)) {
      std::cout << "Processing file '" << dirEntry.path() << "'..." << std::endl;

      Bitmap srcImage = readJpegFile(dirEntry.path());
      Bitmap finalImage = processImage(srcImage);

      saveBitmap(finalImage, outputDir/dirEntry.path().stem().concat(".bmp"));
    }
  }

  return 0;
}
