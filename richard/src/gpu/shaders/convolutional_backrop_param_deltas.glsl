#version 430

#include "common.glsl"

layout(constant_id = 3) const uint KERNEL_W = 1;
layout(constant_id = 4) const uint KERNEL_H = 1;
layout(constant_id = 5) const uint KERNEL_D = 1;
layout(constant_id = 6) const uint FM_W = 1;
layout(constant_id = 7) const uint FM_H = 1;
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

void main() {
  const uint fmIdx = gl_GlobalInvocationID.x;
  const uint k = gl_GlobalInvocationID.y;
  const uint kZ = k / (KERNEL_W * KERNEL_H);
  const uint kY = (k % (KERNEL_W * KERNEL_H)) / KERNEL_W;
  const uint kX = k % KERNEL_W;

  const uint imW = KERNEL_W + FM_W - 1;
  const uint imH = KERNEL_H + FM_H - 1;

  const uint imageOffset = IS_FIRST_LAYER ? Status.sampleIndex * imW * imH : 0;

  float dK = readDeltaK(k);
  float dB = readDeltaB(fmIdx);

  for (uint j = 0; j < FM_H; ++j) {
    for (uint i = 0; i < FM_W; ++i) {
      uint imX = i + kX;
      uint imY = j + kY;

      float delta = READ_D(fmIdx * FM_W * FM_H + j * FM_H + i);

      dK += readImage(imageOffset + imY * imW + imX, kZ) * delta;
      dB += delta;
    }
  }

  writeDeltaK(k, dK);
  writeDeltaB(fmIdx, dB);
}
