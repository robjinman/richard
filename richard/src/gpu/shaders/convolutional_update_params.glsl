#version 430

#include "common.glsl"

layout(constant_id = 3) const uint KERNEL_W = 1;
layout(constant_id = 4) const uint KERNEL_H = 1;
layout(constant_id = 5) const uint KERNEL_D = 1;
layout(constant_id = 6) const float LEARN_RATE = 0.001;
layout(constant_id = 7) const float LEARN_RATE_DECAY = 1.0;

layout(std140, binding = 0) readonly buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) buffer KSsbo {
  vec4 K[];
};

FN_READ(K)
FN_WRITE(K)

layout(std140, binding = 2) buffer BSsbo {
  vec4 B[];
};

FN_READ(B)
FN_WRITE(B)

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
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;
  const uint dIdx = gl_GlobalInvocationID.z;

  const float learnRate = LEARN_RATE * pow(LEARN_RATE_DECAY, Status.epoch);

  const uint kOffset = dIdx * KERNEL_W * KERNEL_H * KERNEL_D;
  const uint kIdx = kOffset + arrayIndex3d(KERNEL_W, KERNEL_H, xIdx, yIdx);
  writeK(kIdx, readK(kIdx) - readDeltaK(kIdx) * learnRate);
  writeDeltaK(kIdx, 0.0);

  if (xIdx == 0 && yIdx == 0) {
    writeB(dIdx, readB(dIdx) - readDeltaB(dIdx) * learnRate);
    writeDeltaB(dIdx, 0.0);
  }
}
