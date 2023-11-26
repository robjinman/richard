#include <output_layer.hpp>
#include <gtest/gtest.h>

class OutputLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(OutputLayerTest, evalForward) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;

  ActivationFn activationFn = [](double x) {
    return x * 2.0;
  };

  ActivationFn activationFnPrime = [](double) {
    return 2.0;
  };

  OutputLayer layer(json, 3);
  layer.setWeights(Matrix({
    { 2, 1, 3 },
    { 1, 4, 2 }
  }));
  layer.setBiases(Vector({ 5, 7 }));
  layer.setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });
  Vector Y(layer.evalForward(X.storage()));

  ASSERT_EQ(Y, Vector({ (3*2+4*1+2*3+5)*2, (3*1+4*4+2*2+7)*2 }));
}

TEST_F(OutputLayerTest, trainForward) {
  // TODO
}
