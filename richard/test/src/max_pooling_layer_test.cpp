#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <max_pooling_layer.hpp>
#include <convolutional_layer.hpp>

class MaxPoolingLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};
/*
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
    9, 8, 7, 6
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

    9, 0, 0, 0,
    0, 0, 8, 0,
    0, 0, 0, 0,
    0, 7, 6, 0
  }));
}

TEST_F(MaxPoolingLayerTest, backpropFromConvLayer_depth1_convDepth1) {
  size_t inputW = 6;
  size_t inputH = 6;
  size_t inputDepth = 1;
  std::array<size_t, 2> regionSize{ 2, 2 };

  nlohmann::json json;
  json["regionSize"] = regionSize;

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  // Each of these elements represents the rate of change of the cost function C with respect to
  // a point in the feature map
  Vector convDelta({
    2.0, 5.0,
    3.0, 4.0
  });

  LayerParams slice0;
  // Each point in the feature map is the result of doing an element-wise multiplication of
  // this matrix with a 2x2 patch of the input pixels and summing the result
  slice0.W = Matrix({
    { 1.0, 2.0 },
    { 4.0, 3.0 }
  });
  slice0.b = 7.0;

  std::vector<LayerParams> convParams({
    slice0
  });

  size_t deltaW = inputW / regionSize[0];
  size_t deltaH = inputH / regionSize[1];

  Vector delta(deltaW * deltaH * inputDepth);

  layer.backpropFromConvLayer(convParams, convDelta, delta);

  ASSERT_EQ(delta, Vector({
    2, 9, 10,
    11, 36, 23,
    12, 25, 12
  }));
}

TEST_F(MaxPoolingLayerTest, backpropFromConvLayer_depth1_convDepth2) {
  size_t inputW = 6;
  size_t inputH = 6;
  size_t inputDepth = 4;
  std::array<size_t, 2> regionSize{ 2, 2 };

  nlohmann::json json;
  json["regionSize"] = regionSize;

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Vector convDelta({
    2.0, 5.0,   // Deltas for feature map 0
    3.0, 4.0,

    3.0, 4.0,   // Deltas for feature map 1
    1.0, 7.0
  });

  LayerParams slice0;
  slice0.W = Matrix({
    { 1.0, 2.0 },
    { 4.0, 3.0 }
  });
  slice0.b = 7.0;

  LayerParams slice1;
  slice1.W = Matrix({
    { 2.0, 3.0 },
    { 5.0, 4.0 }
  });
  slice1.b = 7.0;

  std::vector<LayerParams> convParams({
    slice0,
    slice1
  });

  size_t deltaW = inputW / regionSize[0];
  size_t deltaH = inputH / regionSize[1];

  Vector delta(deltaW * deltaH);

  layer.backpropFromConvLayer(convParams, convDelta, delta);

  Vector fm0Delta({
    2, 9, 10,
    11, 36, 23,
    12, 25, 12
  });

  Vector fm1Delta({
    6, 17, 12,
    17, 49, 37,
    5, 39, 28
  });

  ASSERT_EQ(delta, fm0Delta + fm1Delta);
}
*/