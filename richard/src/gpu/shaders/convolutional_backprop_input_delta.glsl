#version 430

#include "common/common.glsl"

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

FN_WRITE(InputDelta)

// Computes full convolution of every zIdx kernel slice with the delta, repeated for every
// feature map / kernel, and accumulates the results in the input delta.
void main() {
  // One thread for each element of the convolution results
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;
  const uint zIdx = gl_GlobalInvocationID.z;

  // The results of the convolutions are accumulated in the input delta
  const uint resultW = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
  const uint resultH = gl_WorkGroupSize.y * gl_NumWorkGroups.y;

  // The 'image' is the delta, which we're convolving with the kernel
  const uint imW = resultW - KERNEL_W + 1;
  const uint imH = resultH - KERNEL_H + 1;

  const uint inDeltaIdx = arrayIndex3d(resultW, resultH, xIdx, yIdx, zIdx);

  const int xMin = -int(KERNEL_W) + 1;
  const int yMin = -int(KERNEL_H) + 1;

  float sum = 0.0;
  for (uint d = 0; d < NUM_FEATURE_MAPS; ++d) {
    const uint kernelOffset = d * KERNEL_W * KERNEL_H * KERNEL_D;

    for (uint k = 0; k < KERNEL_D; ++k) {

      // Compute a full 2D convolution between the d'th feature map delta and k'th slice of this
      // feature map's associated kernel

      const int jFrom = max(0, int(KERNEL_H) - int(yIdx) - 1);
      const int jTo = min(int(KERNEL_H), int(resultH) - int(yIdx));
      for (int j = jFrom; j < jTo; ++j) {
        const int y = yMin + int(yIdx + j);

        const int iFrom = max(0, int(KERNEL_W) - int(xIdx) - 1);
        const int iTo = min(int(KERNEL_W), int(resultW) - int(xIdx));
        for (int i = iFrom; i < iTo; ++i) {
          const int x = xMin + int(xIdx + i);

          const float pixel = readD(arrayIndex3d(imW, imH, x, y, d));
          const uint kernelIdx = arrayIndex3d(KERNEL_W, KERNEL_H, KERNEL_W - i - 1,
            KERNEL_H - j - 1, k);

          sum += pixel * readK(kernelOffset + kernelIdx);
        }
      }
    }
  }

  writeInputDelta(inDeltaIdx, sum);
}
