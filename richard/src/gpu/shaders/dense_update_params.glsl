#version 430

#include "common.glsl"

layout(constant_id = 3) const uint LAYER_NUM_INPUTS = 1;
layout(constant_id = 4) const float LEARN_RATE = 0.001;
layout(constant_id = 5) const float LEARN_RATE_DECAY = 1.0;

layout(std140, binding = 0) readonly buffer StatusSsbo {
  StatusBuffer Status;
};

layout(std140, binding = 1) buffer BSsbo {
  vec4 B[];
};

FN_READ(B)
FN_WRITE(B)

layout(std140, binding = 2) buffer WSsbo {
  vec4 W[];
};

FN_READ(W)
FN_WRITE(W)

layout(std140, binding = 3) buffer DeltaBSsbo {
  vec4 DeltaB[];
};

FN_READ(DeltaB)
FN_WRITE(DeltaB)

layout(std140, binding = 4) buffer DeltaWSsbo {
  vec4 DeltaW[];
};

FN_READ(DeltaW)
FN_WRITE(DeltaW)

void main() {
  const uint xIdx = gl_GlobalInvocationID.x;
  const uint yIdx = gl_GlobalInvocationID.y;

  const float learnRate = LEARN_RATE * pow(LEARN_RATE_DECAY, Status.epoch);

  const uint wIdx = yIdx * LAYER_NUM_INPUTS + xIdx;

  writeW(wIdx, readW(wIdx) - readDeltaW(wIdx) * learnRate);
  writeDeltaW(wIdx, 0);

  if (xIdx == 0) {
    writeB(yIdx, readB(yIdx) - readDeltaB(yIdx) * learnRate);
    writeDeltaB(yIdx, 0);
  }
}
