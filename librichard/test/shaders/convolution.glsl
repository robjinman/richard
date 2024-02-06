#version 430

#define FN_READ(BUF) \
  float read##BUF(uint pos) { \
    return BUF[pos / 4][pos % 4]; \
  }

#define FN_WRITE(BUF) \
  void write##BUF(uint pos, float val) { \
    BUF[pos / 4][pos % 4] = val; \
  }

uint arrayIndex3d(uint W, uint H, uint x, uint y, uint z) {
  return z * W * H + y * W + x;
}

layout(constant_id = 0) const uint local_size_x = 1;
layout(constant_id = 1) const uint local_size_y = 1;
layout(constant_id = 2) const uint local_size_z = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const uint KERNEL_W = 1;
layout(constant_id = 4) const uint KERNEL_H = 1;
layout(constant_id = 5) const uint KERNEL_D = 1;

layout(std140, binding = 0) readonly buffer ImageSsbo {
  vec4 Image[];
};

FN_READ(Image)

layout(std140, binding = 1) readonly buffer KernelSsbo {
  vec4 Kernel[];
};

FN_READ(Kernel)

layout(std140, binding = 2) writeonly buffer ResultSsbo {
  vec4 Result[];
};

FN_WRITE(Result)

void main() {
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;

  const uint fmW = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
  const uint fmH = gl_WorkGroupSize.y * gl_NumWorkGroups.y;

  const uint imW = fmW + KERNEL_W - 1;
  const uint imH = fmH + KERNEL_H - 1;

  float sum = 0.0;
  for (uint k = 0; k < KERNEL_D; ++k) {
    for (uint j = 0; j < KERNEL_H; ++j) {
      for (uint i = 0; i < KERNEL_W; ++i) {
        const uint x = xIdx + i;
        const uint y = yIdx + j;
        const uint z = k;

        const float pixel = readImage(arrayIndex3d(imW, imH, x, y, z));
        const uint kernelIdx = arrayIndex3d(KERNEL_W, KERNEL_H, KERNEL_W - i - 1,
          KERNEL_H - j - 1, k);
        sum += pixel * readKernel(kernelIdx);
      }
    }
  }

  writeResult(yIdx * fmW + xIdx, sum);
}
