#version 430

#include "common/common.glsl"

layout(constant_id = 3) const uint LAYER_SIZE = 1;
layout(constant_id = 4) const uint LAYER_NUM_INPUTS = 1;

layout(std140, binding = 0) readonly buffer WSsbo {
  vec4 W[];
};

FN_READ(W)

layout(std140, binding = 1) readonly buffer DSsbo {
  vec4 D[];
};

FN_READ(D)

layout(std140, binding = 2) writeonly buffer InputDeltaSsbo {
  vec4 InputDelta[];
};

FN_WRITE(InputDelta)

void main() {
  const uint index = gl_GlobalInvocationID.x;
  const uint inputSize = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  float weightedSum = 0.0;
  for (uint i = 0; i < LAYER_SIZE; ++i) {
    weightedSum += readW(i * inputSize + index) * readD(i);
  }
  writeInputDelta(index, weightedSum);
}
