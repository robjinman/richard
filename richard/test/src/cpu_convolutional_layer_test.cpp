#include "mock_cpu_layer.hpp"
#include <cpu/convolutional_layer.hpp>
#include <gtest/gtest.h>

using namespace richard;
using namespace richard::cpu;

class CpuConvolutionalLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CpuConvolutionalLayerTest, forwardPass_depth1) {
  nlohmann::json json;
  json["depth"] = 1;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ConvolutionalLayer layer(json, { 3, 3, 1 });

  ConvolutionalLayer::Filter filter;
  filter.K = Kernel({{
    { 5, 3 },
    { 1, 2 }
  }});
  filter.b = 7;
  layer.test_setFilters({ filter });

  Array3 inputs({{
    { 0, 1, 2 },
    { 5, 6, 7 },
    { 8, 7, 6 }
  }});

  Array3 Z(2, 2, 1);

  layer.test_forwardPass(inputs, Z);

  Array3 expectedZ(2, 2, 1);
  computeCrossCorrelation(inputs, filter.K, *expectedZ.slice(0));
  expectedZ += filter.b;

  ASSERT_EQ(Z, expectedZ);
}

TEST_F(CpuConvolutionalLayerTest, forwardPass_depth2) {
  nlohmann::json json;
  json["depth"] = 2;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ConvolutionalLayer layer(json, { 3, 3, 1 });

  ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({{
    { 5, 3 },
    { 1, 2 }
  }});
  filter0.b = 7;

  ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({{
    { 8, 4 },
    { 5, 3 }
  }});
  filter1.b = 3;

  layer.test_setFilters({ filter0, filter1 });

  Array3 Z(2, 2, 2);

  Array3 inputs({{
    { 0, 1, 2 },
    { 5, 6, 7 },
    { 8, 7, 6 }
  }});

  Array3 expectedZ(2, 2, 2);
  computeCrossCorrelation(inputs, filter0.K, *expectedZ.slice(0));
  *expectedZ.slice(0) += filter0.b;
  computeCrossCorrelation(inputs, filter1.K, *expectedZ.slice(1));
  *expectedZ.slice(1) += filter1.b;

  layer.test_forwardPass(inputs, Z);

  ASSERT_EQ(Z, expectedZ);
}

TEST_F(CpuConvolutionalLayerTest, forwardPass_inputDepth2_depth2) {
  nlohmann::json json;
  json["depth"] = 2;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ConvolutionalLayer layer(json, { 3, 3, 2 });

  ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({
    {
      { 5, 3 },
      { 1, 2 }
    }, {
      { 8, 4 },
      { 5, 3 }
    }
  });
  filter0.b = 7;

  ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({
    {
      { 5, 3 },
      { 1, 2 }
    }, {
      { 8, 4 },
      { 5, 3 }
    }
  });
  filter1.b = 3;

  layer.test_setFilters({ filter0, filter1 });

  Array3 Z(2, 2, 2);

  Array3 inputs({
    {
      { 0, 1, 2 },
      { 5, 6, 7 },
      { 8, 7, 6 },
    }, {
      { 5, 4, 3 },
      { 2, 1, 0 },
      { 1, 2, 5 }
    }
  });

  Array3 expectedZ(2, 2, 2);
  computeCrossCorrelation(inputs, filter0.K, *expectedZ.slice(0));
  *expectedZ.slice(0) += filter0.b;
  computeCrossCorrelation(inputs, filter1.K, *expectedZ.slice(1));
  *expectedZ.slice(1) += filter1.b;

  layer.test_forwardPass(inputs, Z);

  ASSERT_EQ(Z, expectedZ);
}

TEST_F(CpuConvolutionalLayerTest, updateDelta_inputDepth1_depth2) {
  nlohmann::json json;
  json["depth"] = 2;
  json["kernelSize"] = std::array<size_t, 2>({ 2, 2 });
  json["learnRate"] = 1.0;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ConvolutionalLayer layer(json, { 3, 3, 1 });

  ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({{
    { 5, 3 },
    { 1, 2 }
  }});
  filter0.b = 7;

  ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({{
    { 8, 4 },
    { 5, 3 }
  }});
  filter1.b = 3;

  layer.test_setFilters({ filter0, filter1 });

  Array3 Z(2, 2, 2);

  Array3 poolingLayerDelta({
    {
      { 7 }
    }
  });

  Array3 inputs({{
    { 0, 1, 2 },
    { 5, 6, 7 },
    { 8, 7, 6 }
  }});

  Array3 expectedZ(2, 2, 2);
  computeCrossCorrelation(inputs, filter0.K, *expectedZ.slice(0));
  *expectedZ.slice(0) += filter0.b;
  computeCrossCorrelation(inputs, filter1.K, *expectedZ.slice(1));
  *expectedZ.slice(1) += filter1.b;

  layer.test_forwardPass(inputs, Z);

  ASSERT_EQ(Z, expectedZ);

  Array3 paddedPoolingLayerDelta({
    {
      { 0, 0 },
      { 0, 7 },
    }, {
      { 0, 0 },
      { 0, 7 }
    }
  });

  layer.updateDeltas(inputs.storage(), paddedPoolingLayerDelta.storage());

  // TODO
}

