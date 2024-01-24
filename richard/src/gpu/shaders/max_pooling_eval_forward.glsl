#version 430

#include "common/common.glsl"

layout(constant_id = 3) const uint REGION_W = 1;
layout(constant_id = 4) const uint REGION_H = 1;

layout(std140, binding = 0) readonly buffer XSsbo {
  vec4 X[];
};

FN_READ(X)

layout(std140, binding = 1) writeonly buffer ZSsbo {
  vec4 Z[];
};

FN_WRITE(Z)

void main() {
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;
  const uint zIdx = gl_GlobalInvocationID.z;

  const uint outW = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  const uint outH = gl_NumWorkGroups.y * gl_WorkGroupSize.y;

  const uint imgW = outW * REGION_W;
  const uint imgH = outH * REGION_H;

  float largest = FLOAT_LOWEST;
  uint largestX = 0;
  uint largestY = 0;

  for (uint j = 0; j < REGION_H; ++j) {
    for (uint i = 0; i < REGION_W; ++i) {
      const uint imgX = xIdx * REGION_W + i;
      const uint imgY = yIdx * REGION_H + j;
      const uint imgOffset = arrayIndex3d(imgW, imgH, imgX, imgY, zIdx);

      const float px = readX(imgOffset);

      if (px > largest) {
        largest = px;
        largestX = imgX;
        largestY = imgY;
      }
    }
  }

  const uint outOffset = arrayIndex3d(outW, outH, xIdx, yIdx, zIdx);
  writeZ(outOffset, largest);
}
