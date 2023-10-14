#include <gtest/gtest.h>
#include <max_pooling_layer.hpp>

class MaxPoolingLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(MaxPoolingLayerTest, evalForward_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector inputs({
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 0, 1,
    2, 3, 4, 5
  });

  Vector A = layer.evalForward(inputs);

  ASSERT_EQ(A.size(), 4);

  ASSERT_EQ(A[0], 5);
  ASSERT_EQ(A[1], 7);
  ASSERT_EQ(A[2], 9);
  ASSERT_EQ(A[3], 5);
}

TEST_F(MaxPoolingLayerTest, evalForward_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector inputs({
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 0, 1,
    2, 3, 4, 5,

    6, 7, 8, 9,
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 0, 1
  });

  Vector A = layer.evalForward(inputs);

  ASSERT_EQ(A.size(), 8);

  // First slice
  ASSERT_EQ(A[0], 5);
  ASSERT_EQ(A[1], 7);
  ASSERT_EQ(A[2], 9);
  ASSERT_EQ(A[3], 5);

  // Second slice
  ASSERT_EQ(A[4], 7);
  ASSERT_EQ(A[5], 9);
  ASSERT_EQ(A[6], 9);
  ASSERT_EQ(A[7], 7);
}

TEST_F(MaxPoolingLayerTest, trainForward_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector inputs({
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 0, 1,
    2, 3, 4, 5
  });

  layer.trainForward(inputs);

  Vector A = layer.activations();
  ASSERT_EQ(A.size(), 4);

  ASSERT_EQ(A[0], 5);
  ASSERT_EQ(A[1], 7);
  ASSERT_EQ(A[2], 9);
  ASSERT_EQ(A[3], 5);

  Vector mask = layer.mask();
  ASSERT_EQ(mask, Vector({
    0, 0, 0, 0,
    0, 1, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 1
  }));
}

TEST_F(MaxPoolingLayerTest, trainForward_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector inputs({
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 0, 1,
    2, 3, 4, 5,

    6, 7, 8, 9,
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 0, 1
  });

  layer.trainForward(inputs);

  Vector A = layer.activations();
  ASSERT_EQ(A.size(), 8);

  // First slice
  ASSERT_EQ(A[0], 5);
  ASSERT_EQ(A[1], 7);
  ASSERT_EQ(A[2], 9);
  ASSERT_EQ(A[3], 5);

  // Second slice
  ASSERT_EQ(A[4], 7);
  ASSERT_EQ(A[5], 9);
  ASSERT_EQ(A[6], 9);
  ASSERT_EQ(A[7], 7);

  Vector mask = layer.mask();
  ASSERT_EQ(mask, Vector({
    0, 0, 0, 0,
    0, 1, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 1,

    0, 1, 0, 1,
    0, 0, 0, 0,
    0, 0, 0, 1,
    0, 1, 0, 0
  }));
}

TEST_F(MaxPoolingLayerTest, padDelta_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector delta({
    9, 8, 7, 6
  });

  Vector mask({
    0, 0, 0, 1,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1
  });

  Vector padded(mask.size());

  layer.padDelta(delta, mask, padded);

  ASSERT_EQ(padded, Vector({
    0, 0, 0, 8,
    9, 0, 0, 0,
    0, 7, 0, 0,
    0, 0, 0, 6
  }));
}

TEST_F(MaxPoolingLayerTest, padDelta_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector delta({
    9, 8, 7, 6, 5, 4, 3, 2
  });

  Vector mask({
    0, 0, 0, 1,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,

    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 0,
    0, 1, 1, 0
  });

  Vector padded(mask.size());

  layer.padDelta(delta, mask, padded);

  ASSERT_EQ(padded, Vector({
    0, 0, 0, 8,
    9, 0, 0, 0,
    0, 7, 0, 0,
    0, 0, 0, 6,

    5, 0, 0, 0,
    0, 0, 4, 0,
    0, 0, 0, 0,
    0, 3, 2, 0
  }));
}
