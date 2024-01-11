#version 430

#include "common.glsl"

layout(constant_id = 3) const uint KERNEL_W = 1;
layout(constant_id = 4) const uint KERNEL_H = 1;
layout(constant_id = 5) const uint KERNEL_D = 1;
layout(constant_id = 6) const uint NUM_FEATURE_MAPS = 1;

layout(std140, binding = 0) readonly buffer KSsbo {
  vec4 K[];
};

FN_READ(K)

layout(std140, binding = 1) readonly buffer DSsbo {
  vec4 D[];
};

FN_READ(D)

layout(std140, binding = 2) writeonly buffer InputDeltaSsbo {
  vec4 InputDelta[];
};

FN_READ(InputDelta)
FN_WRITE(InputDelta)

// Computes full convolution of every zIdx kernel slice with the delta, repeated for every
// feature map / kernel, and accumulates the results in the input delta.
void main() {
  // One thread for each element of the convolution results
  const int xIdx = int(gl_GlobalInvocationID.x);
  const int yIdx = int(gl_GlobalInvocationID.y);
  const int zIdx = int(gl_GlobalInvocationID.z);

  // The results of the convolutions are accumulated in the input delta
  const int resultW = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);
  const int resultH = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);

  const int kW = int(KERNEL_W);
  const int kH = int(KERNEL_H);

  // The 'image' is the delta, which we're convolving with the kernel
  const int imW = resultW - kW + 1;
  const int imH = resultH - kH + 1;

  const uint inDeltaIdx = arrayIndex3d(imW, imH, xIdx, yIdx, zIdx);
  writeInputDelta(inDeltaIdx, 0.0);

  const int xMin = -kW + 1;
  const int yMin = -kH + 1;

  for (uint d = 0; d < NUM_FEATURE_MAPS; ++d) {
    const uint kernelOffset = d * KERNEL_W * KERNEL_H * KERNEL_D;

    for (int k = 0; k < KERNEL_D; ++k) {

      // Compute a full 2D convolution between the d'th feature map delta and k'th slice of this
      // feature map's associated kernel

      float sum = 0.0;
      for (int j = max(0, kH - yIdx - 1); j < min(kH, resultH - yIdx); ++j) {
        for (int i = max(0, kW - xIdx - 1); i < min(kW, resultW - xIdx); ++i) {
          const int x = xMin + xIdx + i;
          const int y = yMin + yIdx + j;

          const float pixel = readD(arrayIndex3d(imW, imH, x, y, d));
          const uint kernelIdx = arrayIndex3d(KERNEL_W, KERNEL_H, kW - i - 1, kH - j - 1, k);

          sum += pixel * readK(kernelOffset + kernelIdx);
        }
      }

      writeInputDelta(inDeltaIdx, readInputDelta(inDeltaIdx) + sum);
    }
  }
}
