#version 430

#include "common.glsl"

layout(std140, binding = 0) readonly buffer OutputLayerActivationsSsbo {
  vec4 OutputLayerActivations[];
};

FN_READ(OutputLayerActivations)

layout(std140, binding = 1) readonly buffer YSsbo {
  vec4 Y[];
};

FN_READ(Y)

layout(std140, binding = 2) writeonly buffer CostsSsbo {
  vec4 Costs[];
};

FN_READ(Costs)

void main() {
  const uint index = gl_GlobalInvocationID.x;

  // TODO
}
