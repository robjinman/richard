#include "neural_net.hpp"
#include "utils.hpp"
#include "config.hpp"

namespace richard {

Hyperparams::Hyperparams()
  : epochs(0)
  , batchSize(1000)
  , miniBatchSize(16) {}

Hyperparams::Hyperparams(const Config& config) {
  epochs = config.getValue<size_t>("epochs");
  batchSize = config.getValue<size_t>("batchSize");
  miniBatchSize = config.getValue<size_t>("miniBatchSize");
}

const Config& Hyperparams::exampleConfig() {
  static Config config;
  static bool done = false;

  if (!done) {
    config.setValue("epochs", 10);
    config.setValue("batchSize", 1000);
    config.setValue("miniBatchSize", 16);

    done = true;
  }

  return config;
}

const Config& NeuralNet::exampleConfig() {
  static Config config;
  static bool done = false;

  if (!done) {
    Config layer1;

    layer1.setValue("type", "dense");
    layer1.setValue("size", 300);
    layer1.setValue("learnRate", 0.7);
    layer1.setValue("learnRateDecay", 1.0);
    layer1.setValue("dropoutRate", 0.5);

    Config layer2;
    layer2.setValue("type", "dense");
    layer2.setValue("size", 80);
    layer2.setValue("learnRate", 0.7);
    layer2.setValue("learnRateDecay", 1.0);
    layer2.setValue("dropoutRate", 0.5);

    std::vector<Config> layersConfig{layer1, layer2};

    config.setObject("hyperparams", Hyperparams::exampleConfig());
    config.setObjectArray("hiddenLayers", layersConfig);

    Config outLayer;
    outLayer.setValue("type", "output");
    outLayer.setValue("size", 10);
    outLayer.setValue("learnRate", 0.7);
    outLayer.setValue("learnRateDecay", 1.0);

    config.setObject("outputLayer", outLayer);

    done = true;
  }

  return config;
}

}
