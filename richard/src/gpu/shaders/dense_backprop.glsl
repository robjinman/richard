#version 430

#include "utils.glsl"

layout(constant_id = 3) const uint LAYER_NUM_INPUTS = 1;
layout(constant_id = 4) const uint NEXT_LAYER_SIZE = 1;
layout(constant_id = 5) const bool IS_FIRST_LAYER = false;

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

layout(std140, binding = 4) readonly buffer ZSsbo {
  vec4 Z[];
};

FN_READ(Z)

layout(std140, binding = 5) readonly buffer ASsbo {
  vec4 A[];
};

FN_READ(A)

layout(std140, binding = 6) buffer DSsbo {
  vec4 D[];
};

FN_READ(D)
FN_WRITE(D)

layout(std140, binding = 7) readonly buffer NextWSsbo {
  vec4 NextW[];
};

FN_READ(NextW)

layout(std140, binding = 8) readonly buffer NextDSsbo {
  vec4 NextD[];
};

FN_READ(NextD)

layout(std140, binding = 9) buffer DeltaBSsbo {
  vec4 DeltaB[];
};

FN_READ(DeltaB)
FN_WRITE(DeltaB)

layout(std140, binding = 10) buffer DeltaWSsbo {
  vec4 DeltaW[];
};

FN_READ(DeltaW)
FN_WRITE(DeltaW)

void main() {
  const uint index = gl_GlobalInvocationID.x;

  float weightedSum = 0.0;
  for (uint i = 0; i < NEXT_LAYER_SIZE; ++i) {
    weightedSum += readNextW(i * NEXT_LAYER_SIZE + index) * readNextD(i);
  }
  writeD(index, weightedSum * sigmoidPrime(readZ(index)));

  const uint xOffset = IS_FIRST_LAYER ? Status.sampleIndex * LAYER_NUM_INPUTS : 0;

  for (uint i = 0; i < LAYER_NUM_INPUTS; ++i) {
    uint wIdx = index * LAYER_NUM_INPUTS + i;
    float dw = readDeltaW(wIdx);
    writeDeltaW(wIdx, dw + readX(xOffset + i) * readD(index));
  }

  writeDeltaB(index, readDeltaB(index) + readD(index));
}
