#include "mock_data_loader.hpp"
#include "mock_labelled_data_set.hpp"
#include <richard/cpu/cpu_neural_net.hpp>
#include <richard/cpu/dense_layer.hpp>
#include <richard/cpu/output_layer.hpp>
#include <richard/cpu/convolutional_layer.hpp>
#include <richard/event_system.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace richard;
using namespace richard::cpu;
using testing::NiceMock;

class CpuNeuralNetTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CpuNeuralNetTest, evaluate) {
  const std::string configString =     ""
  "{                                    "
  "  \"hyperparams\": {                 "
  "      \"epochs\": 1,                 "
  "      \"batchSize\": 1,              "
  "      \"miniBatchSize\": 1           "
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

  Size3 inputShape({ 3, 1, 1 });

  auto eventSystem = createEventSystem();

  Config config = Config::fromJson(configString);
  CpuNeuralNetPtr net = createNeuralNet(inputShape, config, *eventSystem);

  std::vector<Sample> samples{Sample{"a", Array3({{{ 0.5f, 0.3f, 0.7f }}})}};

  DataLoaderPtr dataLoader = std::make_unique<MockDataLoader>();
  testing::NiceMock<MockLabelledDataSet> dataSet(std::move(dataLoader),
    std::vector<std::string>({ "a", "b" }));

  ON_CALL(dataSet, loadSamples).WillByDefault(testing::Return(samples));

  Matrix W0({
    { 0.2f, 0.3f, 0.4f },
    { 0.5f, 0.4f, 0.3f },
    { 0.6f, 0.7f, 0.1f },
    { 0.2f, 0.9f, 0.8f }
  });
  Vector B0({ 0.4f, 0.1f, 0.2f, 0.5f });

  Matrix W1({
    { 0.4f, 0.3f, 0.1f, 0.3f },
    { 0.2f, 0.4f, 0.3f, 0.8f },
    { 0.7f, 0.4f, 0.9f, 0.2f },
    { 0.6f, 0.1f, 0.6f, 0.5f },
    { 0.8f, 0.7f, 0.2f, 0.1f }
  });
  Vector B1({ 0.2f, 0.3f, 0.1f, 0.2f, 0.6f });

  Matrix W2({
    { 0.1f, 0.4f, 0.5f, 0.2f, 0.8f },
    { 0.9f, 0.8f, 0.6f, 0.1f, 0.7f }
  });
  Vector B2({ 0.6f, 0.8f });

  dynamic_cast<DenseLayer&>(net->test_getLayer(0)).test_setWeights(W0.storage());
  dynamic_cast<DenseLayer&>(net->test_getLayer(0)).test_setBiases(B0.storage());
  dynamic_cast<DenseLayer&>(net->test_getLayer(1)).test_setWeights(W1.storage());
  dynamic_cast<DenseLayer&>(net->test_getLayer(1)).test_setBiases(B1.storage());
  dynamic_cast<OutputLayer&>(net->test_getLayer(2)).test_setWeights(W2.storage());
  dynamic_cast<OutputLayer&>(net->test_getLayer(2)).test_setBiases(B2.storage());

  net->train(dataSet);
  
  // TODO: Add some assertions
}

TEST_F(CpuNeuralNetTest, evaluateTrivialConvVsFullyConnected) {
  const std::string convNetConfigString =  ""
  "{                                        "
  "  \"hyperparams\": {                     "
  "      \"epochs\": 1,                     "
  "      \"batchSize\": 1,                  "
  "      \"miniBatchSize\": 1               "
  "  },                                     "
  "  \"hiddenLayers\": [                    "
  "      {                                  "
  "          \"type\": \"convolutional\",   "
  "          \"depth\": 1,                  "
  "          \"kernelSize\": [1, 1],        "
  "          \"learnRate\": 0.1,            "
  "          \"learnRateDecay\": 1.0,       "
  "          \"dropoutRate\": 0.0           "
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
  "      \"batchSize\": 1,                  "
  "      \"miniBatchSize\": 1               "
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

  Size3 inputShape({ 2, 2, 1 });

  auto eventSystem = createEventSystem();

  CpuNeuralNetPtr convNet = createNeuralNet(inputShape, Config::fromJson(convNetConfigString),
    *eventSystem);
  CpuNeuralNetPtr denseNet = createNeuralNet(inputShape, Config::fromJson(denseNetConfigString),
    *eventSystem);

  std::vector<Sample> samples{Sample{"a", Array3{{
    { 0.5f, 0.4f },
    { 0.7f, 0.6f },
   }}}};

  DataLoaderPtr dataLoader = std::make_unique<MockDataLoader>();
  testing::NiceMock<MockLabelledDataSet> dataSet(std::move(dataLoader),
    std::vector<std::string>({ "a", "b" }));

  ON_CALL(dataSet, loadSamples).WillByDefault(testing::Return(samples));

  ConvolutionalLayer::Filter filter;
  filter.K = Kernel({
    {
      { 1.f }
    }
  });
  filter.b = 0.0;

  Matrix denseW({
    { 0.1f, 0.4f, 0.3f, 0.2f },
    { 0.9f, 0.3f, 0.1f, 0.5f }
  });
  Vector denseB({ 0.7f, 0.8f });

  Matrix outW({
    { 0.4f, 0.2f },
    { 0.5f, 0.6f }
  });
  Vector outB({ 0.1f, 0.2f });

  dynamic_cast<ConvolutionalLayer&>(convNet->test_getLayer(0)).test_setFilters({ filter });
  dynamic_cast<DenseLayer&>(convNet->test_getLayer(2)).test_setWeights(denseW.storage());
  dynamic_cast<DenseLayer&>(convNet->test_getLayer(2)).test_setBiases(denseB.storage());
  dynamic_cast<OutputLayer&>(convNet->test_getLayer(3)).test_setWeights(outW.storage());
  dynamic_cast<OutputLayer&>(convNet->test_getLayer(3)).test_setBiases(outB.storage());

  dynamic_cast<DenseLayer&>(denseNet->test_getLayer(0)).test_setWeights(denseW.storage());
  dynamic_cast<DenseLayer&>(denseNet->test_getLayer(0)).test_setBiases(denseB.storage());
  dynamic_cast<OutputLayer&>(denseNet->test_getLayer(1)).test_setWeights(outW.storage());
  dynamic_cast<OutputLayer&>(denseNet->test_getLayer(1)).test_setBiases(outB.storage());

  convNet->train(dataSet);
  denseNet->train(dataSet);

  // TODO: Add some assertions
}

TEST_F(CpuNeuralNetTest, evaluateConv) {
  const std::string configString =       ""
  "{                                      "
  "  \"hyperparams\": {                   "
  "      \"epochs\": 1,                   "
  "      \"batchSize\": 1,                "
  "      \"miniBatchSize\": 1             "
  "  },                                   "
  "  \"hiddenLayers\": [                  "
  "      {                                "
  "          \"type\": \"convolutional\", "
  "          \"depth\": 2,                "
  "          \"kernelSize\": [2, 2],      "
  "          \"learnRate\": 0.1,          "
  "          \"learnRateDecay\": 1.0,     "
  "          \"dropoutRate\": 0.0         "
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

  Size3 inputShape({ 5, 5, 1 });

  auto eventSystem = createEventSystem();

  Config config = Config::fromJson(configString);
  CpuNeuralNetPtr net = createNeuralNet(inputShape, config, *eventSystem);

  std::vector<Sample> samples{Sample{"a", Array3{{
    { 0.5f, 0.4f, 0.3f, 0.9f, 0.8f },
    { 0.7f, 0.6f, 0.9f, 0.2f, 0.5f },
    { 0.5f, 0.5f, 0.1f, 0.6f, 0.3f },
    { 0.4f, 0.1f, 0.8f, 0.2f, 0.7f },
    { 0.2f, 0.3f, 0.7f, 0.1f, 0.4f }
   }}}};

  DataLoaderPtr dataLoader = std::make_unique<MockDataLoader>();
  testing::NiceMock<MockLabelledDataSet> dataSet(std::move(dataLoader),
    std::vector<std::string>({ "a", "b" }));

  ON_CALL(dataSet, loadSamples).WillByDefault(testing::Return(samples));

  ConvolutionalLayer::Filter filter0;
  filter0.K = Kernel({
    {
      { 0.5f, 0.3f },
      { 0.1f, 0.2f }
    }
  });
  filter0.b = 0.7f;

  ConvolutionalLayer::Filter filter1;
  filter1.K = Kernel({
    {
      { 0.8f, 0.4f },
      { 0.5f, 0.3f }
    }
  });
  filter1.b = 0.3f;

  Matrix outW({
    { 0.1f, 0.4f, 0.3f, 0.2f, 0.5f, 0.2f, 0.8f, 0.1f },
    { 0.9f, 0.3f, 0.1f, 0.5f, 0.8f, 0.4f, 0.4f, 0.9f }
  });
  Vector outB({ 0.f, 0.f });

  dynamic_cast<ConvolutionalLayer&>(net->test_getLayer(0)).test_setFilters({ filter0, filter1 });
  dynamic_cast<OutputLayer&>(net->test_getLayer(2)).test_setWeights(outW.storage());
  dynamic_cast<OutputLayer&>(net->test_getLayer(2)).test_setBiases(outB.storage());

  net->train(dataSet);
  
  // TODO: Add some assertions
}

