#version 430

#include "common.glsl"

layout(constant_id = 3) const uint REGION_W = 1;
layout(constant_id = 4) const uint REGION_H = 1;

layout(std140, binding = 0) readonly buffer DeltaASsbo {
  vec4 DeltaA[];
};

FN_READ(DeltaA)

layout(std140, binding = 1) readonly buffer MaskSsbo {
  vec4 Mask[];
};

FN_READ(Mask)

layout(std140, binding = 2) writeonly buffer InputDeltaSsbo {
  vec4 InputDelta[];
};

FN_WRITE(InputDelta)

void main() {
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;
  const uint zIdx = gl_GlobalInvocationID.z;

  const uint outW = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  const uint outH = gl_NumWorkGroups.y * gl_WorkGroupSize.y;

  const uint imgW = outW * REGION_W;
  const uint imgH = outH * REGION_H;

  const uint outOffset = arrayIndex3d(outW, outH, xIdx, yIdx, zIdx);

  for (uint j = 0; j < REGION_H; ++j) {
    for (uint i = 0; i < REGION_W; ++i) {
      const uint imgX = xIdx * REGION_W + i;
      const uint imgY = yIdx * REGION_H + j;
      const uint imgOffset = arrayIndex3d(imgW, imgH, imgX, imgY, zIdx);

      writeInputDelta(imgOffset, readDeltaA(outOffset) * readMask(imgOffset));
    }
  }
}
