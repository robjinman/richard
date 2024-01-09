#version 430

#include "common.glsl"

layout(constant_id = 3) const uint DELTA_W = 1;
layout(constant_id = 4) const uint DELTA_H = 1;
layout(constant_id = 5) const uint IMAGE_W = 1;
layout(constant_id = 6) const uint IMAGE_H = 1;
layout(constant_id = 7) const uint IMAGE_D = 1;
layout(constant_id = 8) const bool IS_FIRST_LAYER = false;

layout(std140, binding = 0) readonly buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) readonly buffer ImageSsbo {
  vec4 Image[];
};

FN_READ(Image)

layout(std140, binding = 2) readonly buffer DSsbo {
  vec4 D[];
};

FN_READ(D)

layout(std140, binding = 3) buffer DeltaKSsbo {
  vec4 DeltaK[];
};

FN_READ(DeltaK)
FN_WRITE(DeltaK)

layout(std140, binding = 4) buffer DeltaBSsbo {
  vec4 DeltaB[];
};

FN_READ(DeltaB)
FN_WRITE(DeltaB)

// Compute a cross-correlation between each slice of the layer inputs and each slice of the layer
// delta, accumulating the results in the kernel delta
void main() {
  const uint deltaKW = IMAGE_W - DELTA_W + 1;
  const uint deltaKH = IMAGE_H - DELTA_H + 1;

  const uint xIdx = gl_GlobalInvocationID.x % deltaKW;
  const uint yIdx = gl_GlobalInvocationID.x / deltaKW;
  const uint zIdx = gl_GlobalInvocationID.y;
  const uint dIdx = gl_GlobalInvocationID.z;

  const uint imageOffset = IS_FIRST_LAYER ? Status.sampleIndex * IMAGE_W * IMAGE_H * IMAGE_D : 0;

  float weightedSum = 0.0;
  float sum = 0.0;

  for (uint j = 0; j < DELTA_H; ++j) {
    for (uint i = 0; i < DELTA_W; ++i) {
      const uint x = xIdx + i;
      const uint y = yIdx + j;

      const uint deltaIdx = arrayIndex3d(DELTA_W, DELTA_H, i, j, dIdx);

      const float pixel = readImage(imageOffset + arrayIndex3d(IMAGE_W, IMAGE_H, x, y, zIdx));
      const float deltaValue = readD(deltaIdx);

      weightedSum += pixel * deltaValue;
      sum += deltaValue;
    }
  }

  const uint deltaKOffset = dIdx * deltaKW * deltaKH * IMAGE_D;
  const uint deltaKIdx = deltaKOffset + arrayIndex3d(deltaKW, deltaKH, xIdx, yIdx, zIdx);

  const float dK = readDeltaK(deltaKIdx);

  writeDeltaK(deltaKIdx, dK + weightedSum);

  if (xIdx == 0 && yIdx == 0 && zIdx == 0) {
    writeDeltaB(dIdx, readDeltaB(dIdx) + sum);
  }
}
