#version 430

#include "common.glsl"

layout(constant_id = 3) const uint LAYER_NUM_INPUTS = 1;
layout(constant_id = 4) const uint MINI_BATCH_SIZE = 1;

layout(std140, binding = 0) buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) readonly buffer XSsbo {
  vec4 X[];
};

FN_READ(X)

layout(std140, binding = 2) buffer YSsbo {
  vec4 Y[];
};

FN_READ(Y)
FN_WRITE(Y)

layout(std140, binding = 3) readonly buffer BSsbo {
  vec4 B[];
};

FN_READ(B)

layout(std140, binding = 4) readonly buffer WSsbo {
  vec4 W[];
};

FN_READ(W)

layout(std140, binding = 5) writeonly buffer ZSsbo {
  vec4 Z[];
};

FN_WRITE(Z)

layout(std140, binding = 6) buffer ASsbo {
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

  float diff = readY(index) - readA(index);
  writeY(index, 0.5 * diff * diff); // Reuse Y buffer

  if (index == 0) {
    barrier();

    Status.sampleIndex = (Status.sampleIndex + 1) % MINI_BATCH_SIZE;

    const uint layerSize = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    float cost = 0.0;
    for (uint i = 0; i < layerSize; ++i) {
      cost += readY(i);
    }
    Status.cost = cost;
  }
}
