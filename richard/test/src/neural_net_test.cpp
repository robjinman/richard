#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <neural_net.hpp>
#include <training_data_set.hpp>

class NeuralNetTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

class MockLabelledDataSet : public LabelledDataSet {
  public:
    MockLabelledDataSet(const std::vector<std::string>& labels)
      : LabelledDataSet(labels) {}

    MOCK_METHOD(size_t, loadSamples, (std::vector<Sample>& samples, size_t n), (override));
    MOCK_METHOD(void, seekToBeginning, (), (override));
};

TEST_F(NeuralNetTest, evaluate) {
  const std::string configString =     ""
  "{                                    "
  "  \"hyperparams\": {                 "
  "      \"numInputs\": [3, 1],         "
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

  nlohmann::json config = nlohmann::json::parse(configString);
  std::unique_ptr<NeuralNet> net = createNeuralNet(config);

  Sample sample("a", Array3({{{ 0.5, 0.3, 0.7 }}}));
  auto loadSample = [&sample](std::vector<Sample>& samples, size_t) {
    samples.push_back(sample);
    return 1;
  };

  testing::NiceMock<MockLabelledDataSet> dataSet(std::vector<std::string>({ "a", "b" }));
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
}
/*
TEST_F(NeuralNetTest, evaluateConv) {
  const std::string configString =       ""
  "{                                      "
  "  \"hyperparams\": {                   "
  "      \"numInputs\": [5, 5],           "
  "      \"epochs\": 1,                   "
  "      \"maxBatchSize\": 1              "
  "  },                                   "
  "  \"hiddenLayers\": [                  "
  "      {                                "
  "          \"type\": \"convolutional\", "
  "          \"depth\": 2                 "
  "          \"kernelSize\": [2, 2],      "
  "          \"learnRate\": 0.1,          "
  "          \"learnRateDecay\": 1.0,     "
  "      },                               "
  "      {                                "
  "          \"type\": \"maxPooling\",    "
  "          \"regionSize\": [2, 2],      "
  "      }                                "
  "  ],                                   "
  "  \"outputLayer\": {                   "
  "      \"size\": 2,                     "
  "      \"learnRate\": 0.1,              "
  "      \"learnRateDecay\": 1.0          "
  "  }                                    "
  "}                                      ";

  nlohmann::json config = nlohmann::json::parse(configString);
  std::unique_ptr<NeuralNet> net = createNeuralNet(config);

  Sample sample("a", Array3({{{ 0.5, 0.3, 0.7 }}}));
  auto loadSample = [&sample](std::vector<Sample>& samples, size_t) {
    samples.push_back(sample);
    return 1;
  };

  testing::NiceMock<MockLabelledDataSet> dataSet(std::vector<std::string>({ "a", "b" }));
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

  net->setWeights({ W0, W1, W2 });
  net->setBiases({ B0, B1, B2 });

  net->train(dataSet);
}*/

