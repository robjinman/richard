#include <gtest/gtest.h>
#include <convolutional_layer.hpp>
#include "mock_layer.hpp"

class ConvolutionalLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};
/*
TEST_F(ConvolutionalLayerTest, forwardPass_depth1) {
  nlohmann::json json;
  json["depth"] = 1;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;

  ConvolutionalLayer layer(json, 3, 3, 1);

  Matrix W({
    { 5.0, 3.0 },
    { 1.0, 2.0 }
  });
  layer.setWeights({ W });
  layer.setBiases({ 7.0 });

  auto outputSize = layer.outputSize();
  size_t numOutputs = outputSize[0] * outputSize[1] * outputSize[2];
  Vector Z(numOutputs);

  Vector inputs({
    0, 1, 2,
    5, 6, 7,
    8, 7, 6
  });

  layer.forwardPass(inputs, Z);

  ASSERT_EQ(Z, Vector({
    27.0, 38.0,
    72.0, 77.0
  }));
}

TEST_F(ConvolutionalLayerTest, forwardPass_depth2) {
  nlohmann::json json;
  json["depth"] = 2;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;

  ConvolutionalLayer layer(json, 3, 3, 1);

  Matrix W0({
    { 5.0, 3.0 },
    { 1.0, 2.0 }
  });
  Matrix W1({
    { 8.0, 4.0 },
    { 5.0, 3.0 }
  });
  layer.setWeights({ W0, W1 });
  layer.setBiases({ 7.0, 3.0 });

  auto outputSize = layer.outputSize();
  size_t numOutputs = outputSize[0] * outputSize[1] * outputSize[2];
  Vector Z(numOutputs);

  Vector inputs({
    0, 1, 2,
    5, 6, 7,
    8, 7, 6
  });

  layer.forwardPass(inputs, Z);

  ASSERT_EQ(Z, Vector({
    27.0, 38.0,
    72.0, 77.0,

    50.0, 70.0,
    128.0, 132.0
  }));
}

TEST_F(ConvolutionalLayerTest, forwardPass_inputDepth2_depth2) {
  nlohmann::json json;
  json["depth"] = 2;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;

  ConvolutionalLayer layer(json, 3, 3, 2);

  Matrix W0({
    { 5.0, 3.0 },
    { 1.0, 2.0 }
  });
  Matrix W1({
    { 8.0, 4.0 },
    { 5.0, 3.0 }
  });
  layer.setWeights({ W0, W1 });
  layer.setBiases({ 7.0, 3.0 });

  auto outputSize = layer.outputSize();
  size_t numOutputs = outputSize[0] * outputSize[1] * outputSize[2];
  Vector Z(numOutputs);

  Vector inputs({
    0, 1, 2,
    5, 6, 7,
    8, 7, 6,

    5, 4, 3,
    2, 1, 0,
    1, 2, 5
  });

  layer.forwardPass(inputs, Z);

  ASSERT_EQ(Z, Vector({
    27.0, 38.0,
    72.0, 77.0,

    50.0, 70.0,
    128.0, 132.0,

    48.0, 37.0,
    25.0, 24.0,

    72.0, 52.0,
    34.0, 36.0
  }));
}

TEST_F(ConvolutionalLayerTest, updateDelta_inputDepth1_depth2) {
  nlohmann::json json;
  json["depth"] = 2;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;

  ConvolutionalLayer layer(json, 3, 3, 1);
  MockLayer poolingLayer;

  auto outputSize = layer.outputSize();
  size_t numOutputs = outputSize[0] * outputSize[1] * outputSize[2];
  Vector Z(numOutputs);

  Matrix W0({
    { 5.0, 3.0 },
    { 1.0, 2.0 }
  });
  Matrix W1({
    { 8.0, 4.0 },
    { 5.0, 3.0 }
  });
  layer.setWeights({ W0, W1 });
  layer.setBiases({ 7.0, 3.0 });

  Vector inputs({
    0, 1, 2,
    5, 6, 7,
    8, 7, 6
  });

  Vector poolingLayerDelta({
    7
  });

  layer.forwardPass(inputs, Z);

  ASSERT_EQ(Z, Vector({
    27.0, 38.0,
    72.0, 77.0,

    50.0, 70.0,
    128.0, 132.0
  }));

  Vector paddedPoolingLayerDelta({
    0, 0,
    0, 7,

    0, 0,
    0, 7
  });

  ON_CALL(poolingLayer, delta).WillByDefault(testing::ReturnRef(paddedPoolingLayerDelta));

  layer.updateDelta(inputs, poolingLayer, 0);

  std::cout << "Hello\n";
  std::cout << layer.delta();
}
*/