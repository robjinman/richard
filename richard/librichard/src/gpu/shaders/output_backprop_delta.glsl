#version 430

#include "common/common.glsl"

layout(constant_id = 3) const uint LAYER_NUM_INPUTS = 1;

layout(std140, binding = 0) readonly buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) readonly buffer XSsbo {
  vec4 X[];
};

FN_READ(X)

layout(std140, binding = 2) readonly buffer YSsbo {
  vec4 Y[];
};

FN_READ(Y)

layout(std140, binding = 3) readonly buffer BSsbo {
  vec4 B[];
};

FN_READ(B)

layout(std140, binding = 4) readonly buffer WSsbo {
  vec4 W[];
};

FN_READ(W)

layout(std140, binding = 5) readonly buffer ZSsbo {
  vec4 Z[];
};

FN_READ(Z)

layout(std140, binding = 6) readonly buffer ASsbo {
  vec4 A[];
};

FN_READ(A)

layout(std140, binding = 7) buffer DSsbo {
  vec4 D[];
};

FN_READ(D)
FN_WRITE(D)

layout(std140, binding = 8) buffer DeltaBSsbo {
  vec4 DeltaB[];
};

FN_READ(DeltaB)
FN_WRITE(DeltaB)

layout(std140, binding = 9) buffer DeltaWSsbo {
  vec4 DeltaW[];
};

FN_READ(DeltaW)
FN_WRITE(DeltaW)

void main() {
  const uint index = gl_GlobalInvocationID.x;
  const uint layerSize = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  const uint yOffset = Status.sampleIndex * layerSize;

  const float deltaC = readA(index) - readY(yOffset + index);
  writeD(index, deltaC * sigmoidPrime(readZ(index)));

  for (uint i = 0; i < LAYER_NUM_INPUTS; ++i) {
    const uint wIdx = index * LAYER_NUM_INPUTS + i;
    const float dw = readDeltaW(wIdx);
    writeDeltaW(wIdx, dw + readX(i) * readD(index));
  }

  float db = readDeltaB(index);
  writeDeltaB(index, db + readD(index));
}
