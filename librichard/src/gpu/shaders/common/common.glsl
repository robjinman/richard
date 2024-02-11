#define FLOAT_MIN 1.17549e-38
#define FLOAT_MAX 3.40282e+38
#define FLOAT_LOWEST -3.40282e+38 

#define FN_READ(BUF) \
  float read##BUF(uint pos) { \
    return BUF[pos / 4][pos % 4]; \
  }

#define FN_WRITE(BUF) \
  void write##BUF(uint pos, float val) { \
    BUF[pos / 4][pos % 4] = val; \
  }

struct StatusBuffer {
  uint epoch;
  uint sampleIndex;
};

layout(constant_id = 0) const uint local_size_x = 1;
layout(constant_id = 1) const uint local_size_y = 1;
layout(constant_id = 2) const uint local_size_z = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float sigmoidPrime(float x) {
  float sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
}

float relu(float x) {
  return x < 0.0 ? 0.0 : x;
}

float reluPrime(float x) {
  return x < 0.0 ? 0.0 : 1.0;
}

uint arrayIndex3d(uint W, uint H, uint x, uint y, uint z) {
  return z * W * H + y * W + x;
}

float hash(uint x) {
  x -= (x << 6);
  x ^= (x >> 17);
  x -= (x << 9);
  x ^= (x << 4);
  x -= (x << 3);
  x ^= (x << 10);
  x ^= (x >> 15);
  return float(x) / uint(-1);
}
