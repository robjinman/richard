#include <cstdio>
#include <cstdlib>
#include <jpeglib.h>
#include <iostream>
#include "bitmap.hpp"

using namespace cpputils;

void read_JPEG_file(const char *filename) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPARRAY buffer;
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  FILE *infile = fopen(filename, "rb");
  if (infile == nullptr) {
    std::cerr << "Error opening file: " << filename << std::endl;
    exit(1);
  }
  jpeg_stdio_src(&cinfo, infile);

  jpeg_read_header(&cinfo, TRUE);

  jpeg_start_decompress(&cinfo);
  row_stride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  size_t size[] = { cinfo.output_height, cinfo.output_width, cinfo.output_components };
  Bitmap bm(size);

  size_t j = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);

    for (int i = 0; i < row_stride; i += 3) {
      bm[cinfo.output_height - 1 - j][i / 3][2] = buffer[0][i];
      bm[cinfo.output_height - 1 - j][i / 3][1] = buffer[0][i + 1];
      bm[cinfo.output_height - 1 - j][i / 3][0] = buffer[0][i + 2];
    }
    ++j;
  }

  jpeg_finish_decompress(&cinfo);

  jpeg_destroy_decompress(&cinfo);
  fclose(infile);

  saveBitmap(bm, "out.bmp");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <JPEG file path>" << std::endl;
    return 1;
  }
  read_JPEG_file(argv[1]);
  return 0;
}
