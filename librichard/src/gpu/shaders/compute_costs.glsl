#version 430

#include "common/common.glsl"

layout(constant_id = 3) const uint MINI_BATCH_SIZE = 1;

layout(std140, binding = 0) buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) readonly buffer OutputLayerActivationsSsbo {
  vec4 OutputLayerActivations[];
};

FN_READ(OutputLayerActivations)

layout(std140, binding = 2) readonly buffer YSsbo {
  vec4 Y[];
};

FN_READ(Y)

layout(std140, binding = 3) buffer CostsSsbo {
  vec4 Costs[];
};

FN_READ(Costs)
FN_WRITE(Costs)

void main() {
  const uint index = gl_GlobalInvocationID.x;
  const uint networkOutputSize = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  const uint yOffset = Status.sampleIndex * networkOutputSize;

  float cost = readCosts(index);
  float diff = readY(yOffset + index) - readOutputLayerActivations(index);
  writeCosts(index, cost + 0.5 * diff * diff);

  if (index == 0) {
    Status.sampleIndex = (Status.sampleIndex + 1) % MINI_BATCH_SIZE;
  }
}
