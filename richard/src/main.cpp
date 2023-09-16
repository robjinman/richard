#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <boost/program_options.hpp>
#include "training_data_set.hpp"
#include "test_data_set.hpp"
#include "image_data_loader.hpp"
#include "csv_data_loader.hpp"
#include "classifier.hpp"
#include "exception.hpp"
#include "util.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace po = boost::program_options;

namespace {

const std::string DESCRIPTION = "Richard is gaining power";
const bool NORMALIZE = true; // TODO

void conflictingOptions(const po::variables_map& vm, const std::string& opt1,
  const std::string& opt2) {

  if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) && !vm[opt2].defaulted()) {
    EXCEPTION("Conflicting options '" << opt1 << "' and '" << opt2 << "'.");
  }
}

void optionDependency(const po::variables_map& vm, const std::string& forWhat,
  const std::string& requiredOpt) {

  if (vm.count(forWhat) && !vm[forWhat].defaulted()) {
    if (vm.count(requiredOpt) == 0 || vm[requiredOpt].defaulted()) {
      EXCEPTION("Option '" << forWhat << "' requires option '" << requiredOpt << "'.");
    }
  }
}

void optionChoice(const po::variables_map& vm, const std::vector<std::string>& choices) {
  size_t n = 0;
  for (const auto& choice : choices) {
    n += vm.count(choice);
  }
  if (n != 1) {
    std::stringstream ss;
    ss << "Expected exactly 1 of the following arguments: ";
    for (size_t i = 0; i < choices.size(); ++i) {
      ss << choices[i] << (i + 1 < choices.size() ? "," : ".");
    }
    EXCEPTION(ss.str());
  }
}

void trainClassifier(const std::string& networkFile, const std::string& samplesPath,
  const nlohmann::json& config) {

  std::cout << "Training classifier" << std::endl;

  Classifier classifier(config);

  std::unique_ptr<DataLoader> loader = nullptr;
  if (std::filesystem::is_directory(samplesPath)) {
    loader = std::make_unique<ImageDataLoader>(samplesPath, classifier.classLabels());
  }
  else {
    loader = std::make_unique<CsvDataLoader>(samplesPath, classifier.inputSize());
  }

  auto dataSet = std::make_unique<TrainingDataSet>(std::move(loader), classifier.classLabels(),
    NORMALIZE);

  classifier.train(*dataSet);

  classifier.toFile(networkFile);
}

void testClassifier(const std::string& networkFile, const std::string& samplesPath) {
  Classifier classifier(networkFile);

  std::cout << "Testing classifier" << std::endl;

  std::unique_ptr<DataLoader> loader = nullptr;
  if (std::filesystem::is_directory(samplesPath)) {
    loader = std::make_unique<ImageDataLoader>(samplesPath, classifier.classLabels());
  }
  else {
    loader = std::make_unique<CsvDataLoader>(samplesPath, classifier.inputSize());
  }

  auto dataSet = std::make_unique<TestDataSet>(std::move(loader), classifier.classLabels());
  if (NORMALIZE) {
    dataSet->normalize(classifier.trainingDataStats());
  }

  Classifier::Results results = classifier.test(*dataSet);

  std::cout << "Correct classifications: "
    << results.good << "/" << results.good + results.bad << std::endl;

  std::cout << "Average cost: " << results.cost << std::endl;
}

nlohmann::json loadConfig(const std::string& configFile) {
  std::ifstream f(configFile);
  return nlohmann::json::parse(f);
}

}

// richard --train --samples ../data/ocr/train.csv --config ../data/ocr/config.json --network ../data/ocr/network
// richard --eval --samples ../data/ocr/test.csv --network ../data/ocr/network

// richard --train --samples ../data/catdog/train --config ../data/catdog/config.json --network ../data/catdog/network
// richard --eval --samples ../data/catdog/test --network ../data/catdog/network

int main(int argc, char** argv) {
  try {
    po::options_description desc{DESCRIPTION};
    desc.add_options()
      ("help,h", "Show help")
      ("train,t", "Train a classifier")
      ("eval,e", "Evaluate a classifier with test data")
      ("gen,g", "Generate example neural net config file")
      ("samples,s", po::value<std::string>())
      ("config,c", po::value<std::string>(), "JSON configuration file")
      ("network,n", po::value<std::string>()->required(), "File to save/load neural network state");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    optionChoice(vm, { "train", "eval", "gen" });
    optionDependency(vm, "train", "samples");
    optionDependency(vm, "train", "config");
    optionDependency(vm, "eval", "samples");
    conflictingOptions(vm, "eval", "config");
    conflictingOptions(vm, "gen", "samples");
    conflictingOptions(vm, "gen", "config");
    conflictingOptions(vm, "gen", "network");

    if (vm.count("gen")) {
      nlohmann::json obj;
      obj["classifier"] = Classifier::defaultConfig();
      std::cout << obj.dump(4) << std::endl;
      return 0;
    }

    const bool trainingMode = vm.count("train");
    const std::string networkFile = vm["network"].as<std::string>();
    const std::string samplesPath = vm["samples"].as<std::string>();

    po::notify(vm);

    const auto t1 = high_resolution_clock::now();

    if (trainingMode) {
      std::string configFile = vm["config"].as<std::string>();
      nlohmann::json config = loadConfig(configFile);

      trainClassifier(networkFile, samplesPath, getOrThrow(config, "classifier"));
    }
    else {
      testClassifier(networkFile, samplesPath);
    }

    const auto t2 = high_resolution_clock::now();
    const long long elapsed = duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Running time: " << elapsed << " milliseconds" << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
