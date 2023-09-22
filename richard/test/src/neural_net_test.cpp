#include <gtest/gtest.h>
#include <neural_net.hpp>

class NeuralNetTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(NeuralNetTest, evaluate) {/*
  nlohmann::json config;
  nlohmann::json paramsJson;
  paramsJson["numInputs"] = 1;
  paramsJson["numOutputs"] = 1;
  config["hyperparams"] = paramsJson;
  std::unique_ptr<NeuralNet> net = createNeuralNet(config);

  Matrix W(1, 1);
  W.set(0, 0, 12.3);

  Vector B(1);
  B[0] = 23.4;

  net->setWeights({W});
  net->setBiases({B});

  Vector X({45.6});
  Vector Y = net->evaluate(X);
*/
  //ASSERT_EQ(X[0] * W.at(0, 0) + B[0], Y[0]); // TODO: apply sigmoid
}
