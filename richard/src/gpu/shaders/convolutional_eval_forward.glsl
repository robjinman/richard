#version 430

#include "common/common.glsl"

layout(constant_id = 3) const uint KERNEL_W = 1;
layout(constant_id = 4) const uint KERNEL_H = 1;
layout(constant_id = 5) const uint KERNEL_D = 1;

layout(std140, binding = 0) readonly buffer ImageSsbo {
  vec4 Image[];
};

FN_READ(Image)

layout(std140, binding = 1) readonly buffer KSsbo {
  vec4 K[];
};

FN_READ(K)

layout(std140, binding = 2) readonly buffer BSsbo {
  vec4 B[];
};

FN_READ(B)

layout(std140, binding = 3) writeonly buffer ASsbo {
  vec4 A[];
};

FN_WRITE(A)

void main() {
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;
  const uint zIdx = gl_GlobalInvocationID.z;

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

        const float pixel = readImage(z * imW * imH + y * imW + x);

        const float kernelPixel = readK(
          KERNEL_W * KERNEL_H * KERNEL_D * zIdx +
          KERNEL_W * KERNEL_H * k +
          KERNEL_W * j +
          i
        );

        sum += pixel * kernelPixel;
      }
    }
  }

  sum += readB(zIdx);

  writeA(zIdx * fmW * fmH + yIdx * fmW + xIdx, relu(sum));
}
