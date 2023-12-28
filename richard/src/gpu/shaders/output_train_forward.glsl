#version 430

#include "common.glsl"

layout(constant_id = 3) const uint LAYER_NUM_INPUTS = 1;

layout(std140, binding = 0) readonly buffer XSsbo {
  vec4 X[];
};

FN_READ(X)

layout(std140, binding = 1) readonly buffer BSsbo {
  vec4 B[];
};

FN_READ(B)

layout(std140, binding = 2) readonly buffer WSsbo {
  vec4 W[];
};

FN_READ(W)

layout(std140, binding = 3) writeonly buffer ZSsbo {
  vec4 Z[];
};

FN_WRITE(Z)

layout(std140, binding = 4) buffer ASsbo {
  vec4 A[];
};

FN_READ(A)
FN_WRITE(A)

void main() {
  const uint index = gl_GlobalInvocationID.x;

  float weightedSum = 0.0;
  for (uint i = 0; i < LAYER_NUM_INPUTS; ++i) {
    float w = readW(index * LAYER_NUM_INPUTS + i);
    float x = readX(i);
    weightedSum += w * x;
  }
  weightedSum += readB(index);
  writeZ(index, weightedSum);
  writeA(index, sigmoid(weightedSum));
}
