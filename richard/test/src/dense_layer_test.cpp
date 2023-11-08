#include <gtest/gtest.h>
#include <dense_layer.hpp>
#include "mock_layer.hpp"

class DenseLayerTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(DenseLayerTest, evalForward) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ActivationFn activationFn = [](double x) {
    return x * 2.0;
  };

  ActivationFn activationFnPrime = [](double) {
    return 2.0;
  };

  DenseLayer layer(json, 3);
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

TEST_F(DenseLayerTest, trainForward) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ActivationFn activationFn = [](double x) {
    return x * 2.0;
  };

  ActivationFn activationFnPrime = [](double) {
    return 2.0;
  };

  DenseLayer layer(json, 3);
  layer.setWeights(Matrix({
    { 2, 1, 3 },
    { 1, 4, 2 }
  }));
  layer.setBiases(Vector({ 5, 7 }));
  layer.setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });
  
  layer.trainForward(X.storage());

  ConstVectorPtr pA = Vector::createShallow(layer.activations());
  const Vector& A = *pA;

  Vector expectedZ({ 3*2+4*1+2*3+5, 3*1+4*4+2*2+7 });
  Vector expectedA = expectedZ * 2.0;

  ASSERT_EQ(A, expectedA);
}

TEST_F(DenseLayerTest, updateDelta) {
  nlohmann::json json;
  json["size"] = 2;
  json["learnRate"] = 0.5;
  json["learnRateDecay"] = 1.0;
  json["dropoutRate"] = 0.0;

  ActivationFn activationFn = [](double x) {
    return 0.5 * x * x;
  };

  ActivationFn activationFnPrime = [](double x) {
    return x;
  };

  DenseLayer layer(json, 3);
  layer.setWeights(Matrix({
    { 2, 1, 3 },
    { 1, 4, 2 }
  }));
  layer.setBiases(Vector({ 5, 7 }));
  layer.setActivationFn(activationFn, activationFnPrime);

  Vector X({ 3, 4, 2 });

  layer.trainForward(X.storage());

  Vector expectedZ({ 3*2+4*1+2*3+5, 3*1+4*4+2*2+7 });

  Vector nextDelta({ 2, 3 });
  Matrix nextW({
    { 2, 5 },
    { 4, 1 }
  });

  testing::NiceMock<MockLayer> nextLayer;
  ON_CALL(nextLayer, delta).WillByDefault(testing::ReturnRef(nextDelta.storage()));
  ON_CALL(nextLayer, W).WillByDefault(testing::ReturnRef(nextW));

  layer.updateDelta(X.storage(), nextLayer);

  ConstVectorPtr delta = Vector::createShallow(layer.delta());

  //m_delta = nextLayer.W().transposeMultiply(*pNextDelta)
  //                       .hadamard(m_Z.computeTransform(m_activationFnPrime));

  //ASSERT_EQ(*delta, Vector({ 0, 0, 0 }));
}
