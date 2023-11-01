#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <max_pooling_layer.hpp>
#include <convolutional_layer.hpp>

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

  Array2 inputs({
    { 0, 1, 2, 3 },
    { 4, 5, 6, 7 },
    { 8, 9, 0, 1 },
    { 2, 3, 4, 5 }
  });

  Array2 A(layer.evalForward(inputs.storage()), 2, 2);

  ASSERT_EQ(A, Array2({
    { 5, 7 },
    { 9, 5 }
  }));
}

TEST_F(MaxPoolingLayerTest, evalForward_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array3 inputs({
    {
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 },
      { 2, 3, 4, 5 }
    }, {
      { 6, 7, 8, 9 },
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
    }
  });

  Array3 A(layer.evalForward(inputs.storage()), 2, 2, 2);

  ASSERT_EQ(A, Array3({
    {
      { 5, 7 },
      { 9, 5 }
    }, {
      { 7, 9 },
      { 9, 7 }
    }
  }));
}

TEST_F(MaxPoolingLayerTest, trainForward_1x1_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 1, 1 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array2 inputs({
    { 0, 1, 2, 3 },
    { 4, 5, 6, 7 },
    { 8, 9, 0, 1 },
    { 2, 3, 4, 5 }
  });

  layer.trainForward(inputs.storage());

  Array2 A(layer.activations(), 4, 4);

  ASSERT_EQ(A, Array2({
    { 0, 1, 2, 3 },
    { 4, 5, 6, 7 },
    { 8, 9, 0, 1 },
    { 2, 3, 4, 5 }
  }));

  Array3 mask = layer.mask();

  ASSERT_EQ(mask, Array3({
    {
      { 1, 1, 1, 1 },
      { 1, 1, 1, 1 },
      { 1, 1, 1, 1 },
      { 1, 1, 1, 1 }
    }
  }));
}

TEST_F(MaxPoolingLayerTest, trainForward_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array2 inputs({
    { 0, 1, 2, 3 },
    { 4, 5, 6, 7 },
    { 8, 9, 0, 1 },
    { 2, 3, 4, 5 }
  });

  layer.trainForward(inputs.storage());

  Array2 A(layer.activations(), 2, 2);

  ASSERT_EQ(A, Array2({
    { 5, 7 },
    { 9, 5 }
  }));

  Array3 mask = layer.mask();

  ASSERT_EQ(mask, Array3({
    {
      { 0, 0, 0, 0 },
      { 0, 1, 0, 1 },
      { 0, 1, 0, 0 },
      { 0, 0, 0, 1 }
    }
  }));
}

TEST_F(MaxPoolingLayerTest, trainForward_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array3 inputs({
    {
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 },
      { 2, 3, 4, 5 }
    }, {
      { 6, 7, 8, 9 },
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
    }
  });

  layer.trainForward(inputs.storage());

  Array3 A(layer.activations(), 2, 2, 2);

  ASSERT_EQ(A, Array3({
    {
      { 5, 7 },
      { 9, 5 }
    }, {
      { 7, 9 },
      { 9, 7 }
    }
  }));

  Array3 mask = layer.mask();

  ASSERT_EQ(mask, Array3({
    {
      { 0, 0, 0, 0 },
      { 0, 1, 0, 1 },
      { 0, 1, 0, 0 },
      { 0, 0, 0, 1 }
    }, {
      { 0, 1, 0, 1 },
      { 0, 0, 0, 0 },
      { 0, 0, 0, 1 },
      { 0, 1, 0, 0 }
    }
  }));
}

TEST_F(MaxPoolingLayerTest, padDelta_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array3 delta({{
    { 9, 8 },
    { 7, 6 }
  }});

  Array3 mask({{
    { 0, 0, 0, 1 },
    { 1, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, 0, 0, 1 }
  }});

  Array3 padded(4, 4, 1);

  layer.padDelta(delta, mask, padded);

  ASSERT_EQ(padded, Array3({{
    { 0, 0, 0, 8 },
    { 9, 0, 0, 0 },
    { 0, 7, 0, 0 },
    { 0, 0, 0, 6 }
  }}));
}

TEST_F(MaxPoolingLayerTest, padDelta_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array3 delta({
    {
      { 9, 8 },
      { 7, 6 }
    }, {
      { 5, 1 },
      { 4, 7 }
    }
  });

  Array3 mask({
    {
      { 0, 0, 0, 1 },
      { 1, 0, 0, 0 },
      { 0, 1, 0, 0 },
      { 0, 0, 0, 1 }
    }, {
      { 1, 0, 0, 0 },
      { 0, 0, 1, 0 },
      { 0, 0, 0, 0 },
      { 0, 1, 1, 0 }
    }  
  });

  Array3 padded(4, 4, 2);

  layer.padDelta(delta, mask, padded);

  ASSERT_EQ(padded, Array3({
    {
      { 0, 0, 0, 8 },
      { 9, 0, 0, 0 },
      { 0, 7, 0, 0 },
      { 0, 0, 0, 6 }
    }, {
      { 5, 0, 0, 0 },
      { 0, 0, 1, 0 },
      { 0, 0, 0, 0 },
      { 0, 4, 7, 0 }
    }
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
  Array2 convDelta({
    { 2.0, 5.0 },
    { 3.0, 4.0 }
  });

  ConvolutionalLayer::Filter filter0;
  // Each point in the feature map is the result of doing an element-wise multiplication of
  // this kernel with a 2x2x1 patch of the input pixels and summing the result
  filter0.K = Kernel({{
    { 1.0, 2.0 },
    { 4.0, 3.0 }
  }});
  filter0.b = 7.0;

  std::vector<ConvolutionalLayer::Filter> filters({
    filter0
  });

  size_t deltaW = inputW / regionSize[0];
  size_t deltaH = inputH / regionSize[1];

  Array3 delta(deltaW, deltaH, inputDepth);

  layer.backpropFromConvLayer(filters, convDelta.storage(), delta);

  ASSERT_EQ(delta, Array3({{
    { 2, 9, 10 },
    { 11, 36, 23 },
    { 12, 25, 12 }
  }}));
}

TEST_F(MaxPoolingLayerTest, backpropFromConvLayer_depth1_convDepth2) {
  size_t inputW = 6;
  size_t inputH = 6;
  size_t inputDepth = 1;
  std::array<size_t, 2> regionSize{ 2, 2 };

  nlohmann::json json;
  json["regionSize"] = regionSize;

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array3 convDelta({
    {
      { 2.0, 5.0 },   // Deltas for feature map 0
      { 3.0, 4.0 },
    }, {
      { 3.0, 4.0 },   // Deltas for feature map 1
      { 1.0, 7.0 }
    }
  });

  ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({{
    { 1.0, 2.0 },
    { 4.0, 3.0 }
  }});
  filter0.b = 7.0;

  ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({{
    { 2.0, 3.0 },
    { 5.0, 4.0 }
  }});
  filter1.b = 7.0;

  std::vector<ConvolutionalLayer::Filter> filters({
    filter0,
    filter1
  });

  size_t deltaW = inputW / regionSize[0];
  size_t deltaH = inputH / regionSize[1];

  Array3 delta(deltaW, deltaH, 1);

  layer.backpropFromConvLayer(filters, convDelta.storage(), delta);

  Array2 fm0Delta({ // Delta from kernel/featureMap 0
    { 2, 9, 10 },
    { 11, 36, 23 },
    { 12, 25, 12 }
  });

  Array2 fm1Delta({ // Delta from kernel/featureMap 1
    { 6, 17, 12 },
    { 17, 49, 37 },
    { 5, 39, 28 }
  });

  Array3 expectedDelta((fm0Delta + fm1Delta).storage(), 3, 3, 1);

  ASSERT_EQ(delta, expectedDelta);
}

TEST_F(MaxPoolingLayerTest, backpropFromConvLayer_depth2_convDepth2) {
  size_t inputW = 6;
  size_t inputH = 6;
  size_t inputDepth = 2;
  std::array<size_t, 2> regionSize{ 2, 2 };

  nlohmann::json json;
  json["regionSize"] = regionSize;

  MaxPoolingLayer layer(json, inputW, inputH, inputDepth);

  Array3 convDelta({
    {
      { 2.0, 5.0 },   // Deltas for feature map 0
      { 3.0, 4.0 },
    }, {
      { 3.0, 4.0 },   // Deltas for feature map 1
      { 1.0, 7.0 }
    }
  });

  ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({
    {
      { 1.0, 2.0 },
      { 4.0, 3.0 }
    } , {
      { 5.0, 2.0 },
      { 4.0, 3.0 }
    }
  });
  filter0.b = 7.0;

  ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({
    {
      { 2.0, 3.0 },
      { 5.0, 4.0 }
    }, {
      { 2.0, 3.0 },
      { 5.0, 4.0 }
    }
  });
  filter1.b = 7.0;

  std::vector<ConvolutionalLayer::Filter> filters({
    filter0,
    filter1
  });

  size_t deltaW = inputW / regionSize[0];
  size_t deltaH = inputH / regionSize[1];

  Array3 delta(deltaW, deltaH, 2);

  layer.backpropFromConvLayer(filters, convDelta.storage(), delta);

  Array2 slice0Delta = Array2({
    { 2, 9, 10 },   // Delta from first plane of kernel 0, feature map 0
    { 11, 36, 23 },
    { 12, 25, 12 }
  }) + Array2({
    { 6, 17, 12 },  // Delta from first plane of kernel 1, feature map 1
    { 17, 49, 37 },
    { 5, 39, 28 }
  });

  Array2 slice1Delta = Array2({
    { 10, 29, 10 },  // Delta from second plane of kernel 0, feature map 0
    { 23, 52, 23 },
    { 12, 25, 12 }
  }) + Array2({
    { 6, 17, 12 },  // Delta from second plane of kernel 1, feature map 1
    { 17, 49, 37 },
    { 5, 39, 28 }
  });

  Array3 expectedDelta(DataArray::concat(slice0Delta.storage(), slice1Delta.storage()), 3, 3, 2);

  ASSERT_EQ(delta, expectedDelta);
}

