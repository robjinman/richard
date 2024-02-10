#include "richard/neural_net.hpp"
#include "richard/utils.hpp"
#include "richard/config.hpp"

namespace richard {

const hashedString_t EEpochStarted::name = hashString("epochStarted");
const hashedString_t EEpochCompleted::name = hashString("epochCompleted");
const hashedString_t ESampleProcessed::name = hashString("sampleProcessed");

Hyperparams::Hyperparams()
  : epochs(0)
  , batchSize(1000)
  , miniBatchSize(16) {}

Hyperparams::Hyperparams(const Config& config) {
  epochs = config.getNumber<uint32_t>("epochs");
  batchSize = config.getNumber<uint32_t>("batchSize");
  miniBatchSize = config.getNumber<uint32_t>("miniBatchSize");
}

const Config& Hyperparams::exampleConfig() {
  static Config config;
  static bool done = false;

  if (!done) {
    config.setNumber("epochs", 10);
    config.setNumber("batchSize", 1000);
    config.setNumber("miniBatchSize", 16);

    done = true;
  }

  return config;
}

const Config& NeuralNet::exampleConfig() {
  static Config config;
  static bool done = false;

  if (!done) {
    Config layer1;

    layer1.setString("type", "dense");
    layer1.setNumber("size", 300);
    layer1.setNumber("learnRate", 0.7);
    layer1.setNumber("learnRateDecay", 1.0);
    layer1.setNumber("dropoutRate", 0.5);

    Config layer2;
    layer2.setString("type", "dense");
    layer2.setNumber("size", 80);
    layer2.setNumber("learnRate", 0.7);
    layer2.setNumber("learnRateDecay", 1.0);
    layer2.setNumber("dropoutRate", 0.5);

    std::vector<Config> layersConfig{ layer1, layer2 };

    config.setObject("hyperparams", Hyperparams::exampleConfig());
    config.setObjectArray("hiddenLayers", layersConfig);

    Config outLayer;
    outLayer.setString("type", "output");
    outLayer.setNumber("size", 10);
    outLayer.setNumber("learnRate", 0.7);
    outLayer.setNumber("learnRateDecay", 1.0);

    config.setObject("outputLayer", outLayer);

    done = true;
  }

  return config;
}

}
