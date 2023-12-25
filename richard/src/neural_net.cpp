#include "neural_net.hpp"
#include "util.hpp"

namespace richard {

Hyperparams::Hyperparams()
  : epochs(0)
  , batchSize(1000)
  , miniBatchSize(16) {}

Hyperparams::Hyperparams(const nlohmann::json& obj) {
  epochs = getOrThrow(obj, "epochs").get<size_t>();
  batchSize = getOrThrow(obj, "batchSize").get<size_t>();
  miniBatchSize = getOrThrow(obj, "miniBatchSize").get<size_t>();
}

const nlohmann::json& Hyperparams::exampleConfig() {
  static nlohmann::json obj;
  static bool done = false;

  if (!done) {
    obj["epochs"] = 10;
    obj["batchSize"] = 1000;
    obj["miniBatchSize"] = 16;

    done = true;
  }

  return obj;
}

const nlohmann::json& NeuralNet::exampleConfig() {
  static nlohmann::json config;
  static bool done = false;

  if (!done) {
    nlohmann::json layer1;

    layer1["type"] = "dense";
    layer1["size"] = 300;
    layer1["learnRate"] = 0.7;
    layer1["learnRateDecay"] = 1.0;
    layer1["dropoutRate"] = 0.5;

    nlohmann::json layer2;
    layer2["type"] = "dense";
    layer2["size"] = 80;
    layer2["learnRate"] = 0.7;
    layer2["learnRateDecay"] = 1.0;
    layer2["dropoutRate"] = 0.5;

    std::vector<nlohmann::json> layersJson{layer1, layer2};

    config["hyperparams"] = Hyperparams::exampleConfig();
    config["hiddenLayers"] = layersJson;

    nlohmann::json outLayer;
    outLayer["type"] = "output";
    outLayer["size"] = 10;
    outLayer["learnRate"] = 0.7;
    outLayer["learnRateDecay"] = 1.0;

    config["outputLayer"] = outLayer;

    done = true;
  }

  return config;
}

}
