#version 430

struct StatusBuffer {
  uint epoch;
  float cost;
  uint sampleIndex;
};

layout(constant_id = 0) const uint local_size_x = 1;
layout(constant_id = 1) const uint local_size_y = 1;
layout(constant_id = 2) const uint local_size_z = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(std140, binding = 0) buffer StatusSsbo {
  StatusBuffer Status;
};

void main() {
  const uint index = gl_GlobalInvocationID.x;

  if (index == 0) {
    Status.cost = Status.cost + 123.45;
  }
}
