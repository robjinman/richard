#version 430

#include "common.glsl"

layout(std140, binding = 0) readonly buffer ZSsbo {
  vec4 Z[];
};

FN_READ(Z)

layout(std140, binding = 1) writeonly buffer DSsbo {
  vec4 D[];
};

FN_WRITE(D)

layout(std140, binding = 2) readonly buffer NextDSsbo {
  vec4 NextD[];
};

FN_READ(NextD)

void main() {
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;
  const uint zIdx = gl_GlobalInvocationID.z;

  const uint fmW = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
  const uint fmH = gl_WorkGroupSize.y * gl_NumWorkGroups.y;

  const uint idx =
    zIdx * fmW * fmH +
    yIdx * fmW +
    xIdx;

  writeD(idx, reluPrime(readZ(idx)) * readNextD(idx));
}
