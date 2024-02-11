#version 430

#include "common/common.glsl"

layout(constant_id = 3) const uint LAYER_NUM_INPUTS = 1;
layout(constant_id = 4) const bool IS_FIRST_LAYER = false;
layout(constant_id = 5) const float DROPOUT_RATE = 0.0;

layout(push_constant) uniform PushConstants {
  uint seed;
} constants;

layout(std140, binding = 0) readonly buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) readonly buffer XSsbo {
  vec4 X[];
};

FN_READ(X)

layout(std140, binding = 2) readonly buffer BSsbo {
  vec4 B[];
};

FN_READ(B)

layout(std140, binding = 3) readonly buffer WSsbo {
  vec4 W[];
};

FN_READ(W)

layout(std140, binding = 4) writeonly buffer ZSsbo {
  vec4 Z[];
};

FN_WRITE(Z)

layout(std140, binding = 5) writeonly buffer ASsbo {
  vec4 A[];
};

FN_WRITE(A)

void main() {
  const uint index = gl_GlobalInvocationID.x;
  const bool drop = hash(constants.seed + index) < DROPOUT_RATE;
  const uint xOffset = IS_FIRST_LAYER ? Status.sampleIndex * LAYER_NUM_INPUTS : 0;

  float weightedSum = 0.0;
  for (uint i = 0; i < LAYER_NUM_INPUTS; ++i) {
    float w = readW(index * LAYER_NUM_INPUTS + i);
    float x = readX(xOffset + i);
    weightedSum += w * x;
  }
  weightedSum += readB(index);
  writeZ(index, weightedSum);
  writeA(index, drop ? 0.0 : sigmoid(weightedSum));
}
