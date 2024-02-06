#include "mock_cpu_layer.hpp"
#include <richard/config.hpp>
#include <richard/cpu/dense_layer.hpp>
#include <gtest/gtest.h>

using namespace richard;
using namespace richard::cpu;

class CpuDenseLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CpuDenseLayerTest, evalForward) {
  Config config;
  config.setNumber("size", 2);
  config.setNumber("learnRate", 0.5);
  config.setNumber("learnRateDecay", 1.0);
  config.setNumber("dropoutRate", 0.0);

  ActivationFn activationFn = [](netfloat_t x) {
    return x * 2.f;
  };

  ActivationFn activationFnPrime = [](netfloat_t) {
    return 2.f;
  };

  Matrix W({
    { 2, 1, 3 },
    { 1, 4, 2 }
  });

  Vector B({ 5, 7 });

  DenseLayer layer(config, 3);
  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());
  layer.test_setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });
  Vector Y(layer.evalForward(X.storage()));

  ASSERT_EQ(Y, Vector({ (3*2+4*1+2*3+5)*2, (3*1+4*4+2*2+7)*2 }));
}

TEST_F(CpuDenseLayerTest, trainForward) {
  Config config;
  config.setNumber("size", 2);
  config.setNumber("learnRate", 0.5);
  config.setNumber("learnRateDecay", 1.0);
  config.setNumber("dropoutRate", 0.0);

  ActivationFn activationFn = [](netfloat_t x) {
    return x * 2.f;
  };

  ActivationFn activationFnPrime = [](netfloat_t) {
    return 2.f;
  };

  Matrix W({
    { 2, 1, 3 },
    { 1, 4, 2 }
  });

  Vector B({ 5, 7 });

  DenseLayer layer(config, 3);
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
  Config config;
  config.setNumber("size", 2);
  config.setNumber("learnRate", 0.5);
  config.setNumber("learnRateDecay", 1.0);
  config.setNumber("dropoutRate", 0.0);

  ActivationFn activationFn = [](netfloat_t x) {
    return 0.f * x * x;
  };

  ActivationFn activationFnPrime = [](netfloat_t x) {
    return x;
  };

  Matrix W({
    { 2, 1, 3 },
    { 1, 4, 2 }
  });

  Vector B({ 5, 7 });

  DenseLayer layer(config, 3);
  layer.test_setWeights(W.storage());
  layer.test_setBiases(B.storage());
  layer.test_setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });

  layer.trainForward(X.storage());

  Vector dA({ 2, 3 });

  Vector expectedZ({ 3*2+4*1+2*3+5, 3*1+4*4+2*2+7 });
  Vector expectedDelta = dA.hadamard(expectedZ.computeTransform(activationFnPrime));
  Vector expectedDeltaInputs = W.transposeMultiply(expectedDelta);

  layer.updateDeltas(X.storage(), dA.storage());

  ConstVectorPtr dInputs = Vector::createShallow(layer.inputDelta());

  ASSERT_EQ(*dInputs, expectedDeltaInputs);
}
