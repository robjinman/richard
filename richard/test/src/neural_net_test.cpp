#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <neural_net.hpp>
#include <labelled_data_set.hpp>
#include "mock_logger.hpp"

using testing::NiceMock;

class NeuralNetTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

class MockDataLoader : public DataLoader {
  public:
    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples, size_t n), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

class MockLabelledDataSet : public LabelledDataSet {
  public:
    MockLabelledDataSet(DataLoaderPtr dataLoader, const std::vector<std::string>& labels)
      : LabelledDataSet(std::move(dataLoader), labels) {}

    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples, size_t n), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

TEST_F(NeuralNetTest, evaluate) {
  const std::string configString =     ""
  "{                                    "
  "  \"hyperparams\": {                 "
  "      \"epochs\": 1,                 "
  "      \"maxBatchSize\": 1            "
  "  },                                 "
  "  \"hiddenLayers\": [                "
  "      {                              "
  "          \"type\": \"dense\",       "
  "          \"size\": 4,               "
  "          \"learnRate\": 0.1,        "
  "          \"learnRateDecay\": 1.0,   "
  "          \"dropoutRate\": 0.0       "
  "      },                             "
  "      {                              "
  "          \"type\": \"dense\",       "
  "          \"size\": 5,               "
  "          \"learnRate\": 0.1,        "
  "          \"learnRateDecay\": 1.0,   "
  "          \"dropoutRate\": 0.0       "
  "      }                              "
  "  ],                                 "
  "  \"outputLayer\": {                 "
  "      \"size\": 2,                   "
  "      \"learnRate\": 0.1,            "
  "      \"learnRateDecay\": 1.0        "
  "  }                                  "
  "}                                    ";

  NiceMock<MockLogger> logger;

  Triple inputShape({ 3, 1, 1 });

  nlohmann::json config = nlohmann::json::parse(configString);
  std::unique_ptr<NeuralNet> net = createNeuralNet(inputShape, config, logger);

  Sample sample("a", Array3({{{ 0.5, 0.3, 0.7 }}}));
  auto loadSample = [&sample](std::vector<Sample>& samples, size_t) {
    samples.push_back(sample);
    return 1;
  };

  DataLoaderPtr dataLoader = std::make_unique<MockDataLoader>();
  testing::NiceMock<MockLabelledDataSet> dataSet(std::move(dataLoader),
    std::vector<std::string>({ "a", "b" }));

  ON_CALL(dataSet, loadSamples).WillByDefault(testing::Invoke(loadSample));

  Matrix W0({
    { 0.2, 0.3, 0.4 },
    { 0.5, 0.4, 0.3 },
    { 0.6, 0.7, 0.1 },
    { 0.2, 0.9, 0.8 }
  });
  Vector B0({ 0.4, 0.1, 0.2, 0.5 });

  Matrix W1({
    { 0.4, 0.3, 0.1, 0.3 },
    { 0.2, 0.4, 0.3, 0.8 },
    { 0.7, 0.4, 0.9, 0.2 },
    { 0.6, 0.1, 0.6, 0.5 },
    { 0.8, 0.7, 0.2, 0.1 }
  });
  Vector B1({ 0.2, 0.3, 0.1, 0.2, 0.6 });

  Matrix W2({
    { 0.1, 0.4, 0.5, 0.2, 0.8 },
    { 0.9, 0.8, 0.6, 0.1, 0.7 }
  });
  Vector B2({ 0.6, 0.8 });

  net->setWeights({ { W0.storage() }, { W1.storage() }, { W2.storage() } });
  net->setBiases({ B0.storage(), B1.storage(), B2.storage() });

  net->train(dataSet);
  
  // TODO: Add some assertions
}

// In theory, a convolutional layer with a kernel of size of 1x1 (and value 1.0) and a bias of 0.0
// should have no effect. Likewise, a max pooling layer with region size of 1x1 will have no effect.
// Here we compare the behaviour of a convnet containing these "dummy" layers with a fully connected
// network and assert that they are essentially identical.
TEST_F(NeuralNetTest, evaluateTrivialConvVsFullyConnected) {
  const std::string convNetConfigString =  ""
  "{                                        "
  "  \"hyperparams\": {                     "
  "      \"epochs\": 1,                     "
  "      \"maxBatchSize\": 1                "
  "  },                                     "
  "  \"hiddenLayers\": [                    "
  "      {                                  "
  "          \"type\": \"convolutional\",   "
  "          \"depth\": 1,                  "
  "          \"kernelSize\": [1, 1],        "
  "          \"learnRate\": 0.1,            "
  "          \"learnRateDecay\": 1.0        "
  "      },                                 "
  "      {                                  "
  "          \"type\": \"maxPooling\",      "
  "          \"regionSize\": [1, 1]         "
  "      },                                 "
  "      {                                  "
  "          \"type\": \"dense\",           "
  "          \"size\": 2,                   "
  "          \"learnRate\": 0.1,            "
  "          \"learnRateDecay\": 1.0,       "
  "          \"dropoutRate\": 0.0           "
  "      }                                  "
  "  ],                                     "
  "  \"outputLayer\": {                     "
  "      \"size\": 2,                       "
  "      \"learnRate\": 0.1,                "
  "      \"learnRateDecay\": 1.0            "
  "  }                                      "
  "}                                        ";

  const std::string denseNetConfigString = ""
  "{                                        "
  "  \"hyperparams\": {                     "
  "      \"epochs\": 1,                     "
  "      \"maxBatchSize\": 1                "
  "  },                                     "
  "  \"hiddenLayers\": [                    "
  "      {                                  "
  "          \"type\": \"dense\",           "
  "          \"size\": 2,                   "
  "          \"learnRate\": 0.1,            "
  "          \"learnRateDecay\": 1.0,       "
  "          \"dropoutRate\": 0.0           "
  "      }                                  "
  "  ],                                     "
  "  \"outputLayer\": {                     "
  "      \"size\": 2,                       "
  "      \"learnRate\": 0.1,                "
  "      \"learnRateDecay\": 1.0            "
  "  }                                      "
  "}                                        ";

  NiceMock<MockLogger> logger;

  Triple inputShape({ 2, 2, 1 });

  NeuralNetPtr convNet = createNeuralNet(inputShape, nlohmann::json::parse(convNetConfigString),
    logger);
  NeuralNetPtr denseNet = createNeuralNet(inputShape, nlohmann::json::parse(denseNetConfigString),
    logger);

  Sample sample("a", Array3({{
    { 0.5, 0.4 },
    { 0.7, 0.6 },
   }}));
  auto loadSample = [&sample](std::vector<Sample>& samples, size_t) {
    samples.push_back(sample);
    return 1;
  };

  DataLoaderPtr dataLoader = std::make_unique<MockDataLoader>();
  testing::NiceMock<MockLabelledDataSet> dataSet(std::move(dataLoader),
    std::vector<std::string>({ "a", "b" }));

  ON_CALL(dataSet, loadSamples).WillByDefault(testing::Invoke(loadSample));

  Kernel convK({
    {
      { 1.0 }
    }
  });
  Vector convB({ 0.0 });

  Matrix denseW({
    { 0.1, 0.4, 0.3, 0.2 },
    { 0.9, 0.3, 0.1, 0.5 }
  });
  Vector denseB({ 0.7, 0.8 });

  Matrix outW({
    { 0.4, 0.2 },
    { 0.5, 0.6 }
  });
  Vector outB({ 0.1, 0.2 });

  convNet->setWeights({ { convK.storage() }, {}, { denseW.storage() }, { outW.storage() } });
  convNet->setBiases({ convB.storage(), {}, denseB.storage(), outB.storage() });

  denseNet->setWeights({ { denseW.storage() }, { outW.storage() } });
  denseNet->setBiases({ denseB.storage(), outB.storage() });

  convNet->train(dataSet);
  denseNet->train(dataSet);

  // TODO: Add some assertions
}

TEST_F(NeuralNetTest, evaluateConv) {
  const std::string configString =       ""
  "{                                      "
  "  \"hyperparams\": {                   "
  "      \"epochs\": 1,                   "
  "      \"maxBatchSize\": 1              "
  "  },                                   "
  "  \"hiddenLayers\": [                  "
  "      {                                "
  "          \"type\": \"convolutional\", "
  "          \"depth\": 2,                "
  "          \"kernelSize\": [2, 2],      "
  "          \"learnRate\": 0.1,          "
  "          \"learnRateDecay\": 1.0      "
  "      },                               "
  "      {                                "
  "          \"type\": \"maxPooling\",    "
  "          \"regionSize\": [2, 2]       "
  "      }                                "
  "  ],                                   "
  "  \"outputLayer\": {                   "
  "      \"size\": 2,                     "
  "      \"learnRate\": 0.1,              "
  "      \"learnRateDecay\": 1.0          "
  "  }                                    "
  "}                                      ";

  NiceMock<MockLogger> logger;

  Triple inputShape({ 5, 5, 1 });

  nlohmann::json config = nlohmann::json::parse(configString);
  std::unique_ptr<NeuralNet> net = createNeuralNet(inputShape, config, logger);

  Sample sample("a", Array3({{
    { 0.5, 0.4, 0.3, 0.9, 0.8 },
    { 0.7, 0.6, 0.9, 0.2, 0.5 },
    { 0.5, 0.5, 0.1, 0.6, 0.3 },
    { 0.4, 0.1, 0.8, 0.2, 0.7 },
    { 0.2, 0.3, 0.7, 0.1, 0.4 }
   }}));
  auto loadSample = [&sample](std::vector<Sample>& samples, size_t) {
    samples.push_back(sample);
    return 1;
  };

  DataLoaderPtr dataLoader = std::make_unique<MockDataLoader>();
  testing::NiceMock<MockLabelledDataSet> dataSet(std::move(dataLoader),
    std::vector<std::string>({ "a", "b" }));

  ON_CALL(dataSet, loadSamples).WillByDefault(testing::Invoke(loadSample));

  Kernel convK0({
    {
      { 0.5, 0.3 },
      { 0.1, 0.2 }
    }
  });
  Kernel convK1({
    {
      { 0.8, 0.4 },
      { 0.5, 0.3 }
    }
  });
  Vector convB({ 0.7, 0.3 });

  Matrix outW({
    { 0.1, 0.4, 0.3, 0.2, 0.5, 0.2, 0.8, 0.1 },
    { 0.9, 0.3, 0.1, 0.5, 0.8, 0.4, 0.4, 0.9 }
  });
  Vector outB({ 0.0, 0.0 });

  net->setWeights({ { convK0.storage(), convK1.storage() }, {}, { outW.storage() } });
  net->setBiases({ convB.storage(), {}, outB.storage() });
  
  net->train(dataSet);
  
  // TODO: Add some assertions
}

