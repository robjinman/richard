#include <cstdio>
#include <cstdlib>
#include <jpeglib.h>
#include <iostream>

void read_JPEG_file(const char *filename) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPARRAY buffer;
  int row_stride;

  // Step 1: Allocate and initialize JPEG decompression object
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // Step 2: Specify data source (e.g., a file)
  FILE *infile = fopen(filename, "rb");
  if (infile == nullptr) {
    std::cerr << "Error opening file: " << filename << std::endl;
    exit(1);
  }
  jpeg_stdio_src(&cinfo, infile);

  // Step 3: Read file parameters with jpeg_read_header()
  jpeg_read_header(&cinfo, TRUE);

  // Step 4: Start decompressor
  jpeg_start_decompress(&cinfo);
  row_stride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  // Step 5: While scan lines remain to be read, read them.
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);
    // Process the scanline here.
    // For example, print out the RGB values.
    for (int i = 0; i < row_stride; i += 3) {
        printf("(%d, %d, %d) ", buffer[0][i], buffer[0][i + 1], buffer[0][i + 2]);
    }
    printf("\n");
  }

  // Step 6: Finish decompression
  jpeg_finish_decompress(&cinfo);

  // Step 7: Release JPEG decompression object
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <JPEG file path>\n";
    return 1;
  }
  read_JPEG_file(argv[1]);
  return 0;
}
