#include "mock_cpu_layer.hpp"
#include <config.hpp>
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
  Config config;
  config.setInteger("depth", 1);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", 1.0);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  ConvolutionalLayer layer(config, { 3, 3, 1 });

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

  layer.trainForward(inputs.storage());

  Array3 expectedZ(2, 2, 1);
  computeCrossCorrelation(inputs, filter.K, *expectedZ.slice(0));
  expectedZ += filter.b;

  Array3 A(layer.activations(), 2, 2, 1);

  ASSERT_EQ(A, expectedZ.computeTransform(relu));
}

TEST_F(CpuConvolutionalLayerTest, forwardPass_depth2) {
  Config config;
  config.setInteger("depth", 2);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", 1.0);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  ConvolutionalLayer layer(config, { 3, 3, 1 });

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

  layer.trainForward(inputs.storage());

  Array3 A(layer.activations(), 2, 2, 2);

  ASSERT_EQ(A, expectedZ.computeTransform(relu));
}

TEST_F(CpuConvolutionalLayerTest, forwardPass_inputDepth2_depth2) {
  Config config;
  config.setInteger("depth", 2);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", 1.0);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  ConvolutionalLayer layer(config, { 3, 3, 2 });

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

  layer.trainForward(inputs.storage());

  Array3 A(layer.activations(), 2, 2, 2);

  ASSERT_EQ(A, expectedZ.computeTransform(relu));
}

TEST_F(CpuConvolutionalLayerTest, updateDelta_inputDepth1_depth2) {
  Config config;
  config.setInteger("depth", 2);
  config.setIntegerArray<size_t>("kernelSize", { 2, 2 });
  config.setFloat("learnRate", 1.0);
  config.setFloat("learnRateDecay", 1.0);
  config.setFloat("dropoutRate", 0.0);

  ConvolutionalLayer layer(config, { 3, 3, 1 });

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

  layer.trainForward(inputs.storage());

  Array3 A(layer.activations(), 2, 2, 2);

  ASSERT_EQ(A, expectedZ.computeTransform(relu));

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

