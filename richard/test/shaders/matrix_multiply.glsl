#version 430

#define FN_READ(BUF) \
  float read##BUF(uint pos) { \
    return BUF[pos / 4][pos % 4]; \
  }

#define FN_WRITE(BUF) \
  void write##BUF(uint pos, float val) { \
    BUF[pos / 4][pos % 4] = val; \
  }

layout(constant_id = 0) const uint local_size_x = 1;
layout(constant_id = 1) const uint local_size_y = 1;
layout(constant_id = 2) const uint local_size_z = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const uint VECTOR_SIZE = 1;

layout(std140, binding = 0) readonly buffer MSsbo {
  vec4 M[];
};

FN_READ(M)

layout(std140, binding = 1) readonly buffer VSsbo {
  vec4 V[];
};

FN_READ(V)

layout(std140, binding = 2) writeonly buffer RSsbo {
  vec4 R[];
};

FN_WRITE(R)

void main() {
  const uint index = gl_GlobalInvocationID.x;

  float weightedSum = 0.0;
  for (uint i = 0; i < VECTOR_SIZE; ++i) {
    float m = readM(index * VECTOR_SIZE + i);
    float v = readV(i);
    weightedSum += m * v;
  }
  writeR(index, weightedSum);
}
