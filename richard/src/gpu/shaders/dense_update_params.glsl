#version 430

#include "utils.glsl"

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

layout(std140, binding = 3) readonly buffer DeltaBSsbo {
  vec4 DeltaB[];
};

FN_READ(DeltaB)

layout(std140, binding = 4) readonly buffer DeltaWSsbo {
  vec4 DeltaW[];
};

FN_READ(DeltaW)

void main() {
  const uint index = gl_GlobalInvocationID.x;
  const float learnRate = LEARN_RATE * pow(LEARN_RATE_DECAY, Status.epoch);

  for (uint i = 0; i < LAYER_NUM_INPUTS; ++i) {
    uint wIdx = index * LAYER_NUM_INPUTS + i;
    float w = readW(wIdx);
    float dw = readDeltaW(wIdx);
    writeW(wIdx, w - dw * learnRate);
  }

  writeB(index, readB(index) - readDeltaB(index) * learnRate);
}

