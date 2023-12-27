#include <cpu/output_layer.hpp>
#include <gtest/gtest.h>

using namespace richard;
using namespace richard::cpu;

class CpuOutputLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CpuOutputLayerTest, evalForward) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;

  ActivationFn activationFn = [](netfloat_t x) {
    return x * 2.0;
  };

  ActivationFn activationFnPrime = [](netfloat_t) {
    return 2.0;
  };

  Matrix W({
    { 2, 1, 3 },
    { 1, 4, 2 }
  });

  Vector B({ 5, 7 });

  OutputLayer layer(json, 3);
  layer.setWeights(W.storage());
  layer.setBiases(B.storage());
  layer.setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });
  Vector Y(layer.evalForward(X.storage()));

  ASSERT_EQ(Y, Vector({ (3*2+4*1+2*3+5)*2, (3*1+4*4+2*2+7)*2 }));
}

TEST_F(CpuOutputLayerTest, trainForward) {
  // TODO
}
