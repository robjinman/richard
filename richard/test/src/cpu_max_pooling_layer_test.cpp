#include <cpu/max_pooling_layer.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace richard;
using namespace richard::cpu;

class CpuMaxPoolingLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CpuMaxPoolingLayerTest, evalForward_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

TEST_F(CpuMaxPoolingLayerTest, evalForward_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

TEST_F(CpuMaxPoolingLayerTest, trainForward_1x1_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 1, 1 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

  Array3 mask = layer.test_mask();

  ASSERT_EQ(mask, Array3({
    {
      { 1, 1, 1, 1 },
      { 1, 1, 1, 1 },
      { 1, 1, 1, 1 },
      { 1, 1, 1, 1 }
    }
  }));
}

TEST_F(CpuMaxPoolingLayerTest, trainForward_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

  Array3 mask = layer.test_mask();

  ASSERT_EQ(mask, Array3({
    {
      { 0, 0, 0, 0 },
      { 0, 1, 0, 1 },
      { 0, 1, 0, 0 },
      { 0, 0, 0, 1 }
    }
  }));
}

TEST_F(CpuMaxPoolingLayerTest, trainForward_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

  Array3 mask = layer.test_mask();

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

TEST_F(CpuMaxPoolingLayerTest, updateDeltas_depth1) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 1;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

  layer.test_setMask(mask);
  layer.updateDeltas(DataArray(), delta.storage());

  const DataArray& paddedDelta = layer.inputDelta();

  ConstArray3Ptr pPadded = Array3::createShallow(paddedDelta, 4, 4, 1);

  ASSERT_EQ(*pPadded, Array3({{
    { 0, 0, 0, 8 },
    { 9, 0, 0, 0 },
    { 0, 7, 0, 0 },
    { 0, 0, 0, 6 }
  }}));
}

TEST_F(CpuMaxPoolingLayerTest, updateDeltas_depth2) {
  size_t inputW = 4;
  size_t inputH = 4;
  size_t inputDepth = 2;

  nlohmann::json json;
  json["regionSize"] = std::array<size_t, 2>{ 2, 2 };

  MaxPoolingLayer layer(json, { inputW, inputH, inputDepth });

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

  layer.test_setMask(mask);
  layer.updateDeltas(DataArray(), delta.storage());

  const DataArray& paddedDelta = layer.inputDelta();

  ConstArray3Ptr pPadded = Array3::createShallow(paddedDelta, 4, 4, 2);

  ASSERT_EQ(*pPadded, Array3({
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
