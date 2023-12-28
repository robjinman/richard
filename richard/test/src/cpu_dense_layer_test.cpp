#include "mock_cpu_layer.hpp"
#include <cpu/dense_layer.hpp>
#include <gtest/gtest.h>

using namespace richard;
using namespace richard::cpu;

class CpuDenseLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CpuDenseLayerTest, evalForward) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

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

  DenseLayer layer(json, 3);
  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());
  layer.test_setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });
  Vector Y(layer.evalForward(X.storage()));

  ASSERT_EQ(Y, Vector({ (3*2+4*1+2*3+5)*2, (3*1+4*4+2*2+7)*2 }));
}

TEST_F(CpuDenseLayerTest, trainForward) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

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

  DenseLayer layer(json, 3);
  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());
  layer.test_setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });
  
  layer.trainForward(X.storage());

  ConstVectorPtr pA = Vector::createShallow(layer.activations());
  const Vector& A = *pA;

  Vector expectedZ({ 3*2+4*1+2*3+5, 3*1+4*4+2*2+7 });
  Vector expectedA = expectedZ * 2.0;

  ASSERT_EQ(A, expectedA);
}

TEST_F(CpuDenseLayerTest, updateDelta) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ActivationFn activationFn = [](netfloat_t x) {
    return 0.5 * x * x;
  };

  ActivationFn activationFnPrime = [](netfloat_t x) {
    return x;
  };

  Matrix W({
    { 2, 1, 3 },
    { 1, 4, 2 }
  });

  Vector B({ 5, 7 });

  DenseLayer layer(json, 3);
  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());
  layer.test_setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });

  layer.trainForward(X.storage());

  Vector nextDelta({ 2, 3 });
  Matrix nextW({
    { 2, 5 },
    { 4, 1 }
  });

  testing::NiceMock<MockCpuLayer> nextLayer;
  ON_CALL(nextLayer, delta).WillByDefault(testing::ReturnRef(nextDelta.storage()));
  ON_CALL(nextLayer, W).WillByDefault(testing::ReturnRef(nextW));

  layer.updateDelta(X.storage(), nextLayer);

  ConstVectorPtr delta = Vector::createShallow(layer.delta());

  ASSERT_EQ(*delta, Vector({ 336, 390 }));
}
