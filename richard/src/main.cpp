#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include "csv.hpp"
#include "classifier.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace po = boost::program_options;

namespace {

const std::string DESCRIPTION = "Richard is gaining power";

void conflictingOptions(const po::variables_map& vm, const std::string& opt1,
  const std::string& opt2) {

  if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) && !vm[opt2].defaulted()) {
    throw std::logic_error(std::string("Conflicting options '") + opt1 + "' and '" + opt2 + "'.");
  }
}

void optionDependency(const po::variables_map& vm, const std::string& forWhat,
  const std::string& requiredOpt) {

  if (vm.count(forWhat) && !vm[forWhat].defaulted()) {
    if (vm.count(requiredOpt) == 0 || vm[requiredOpt].defaulted()) {
      throw std::logic_error(std::string("Option '") + forWhat
        + "' requires option '" + requiredOpt + "'.");
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
    throw std::logic_error(ss.str());
  }
}

}

// richard --train --samples ../data/ocr/train.csv --config ../data/ocr/netconfig.txt --labels 0 1 2 3 4 5 6 7 8 9 --network ../data/ocr/network
// richard --eval --samples ../data/ocr/test.csv --network ../data/ocr/network

int main(int argc, char** argv) {
  try {
    po::options_description desc{DESCRIPTION};
    desc.add_options()
      ("help,h", "Show help")
      ("train,t", "Train a classifier")
      ("eval,e", "Evaluate a classifier with test data")
      ("gen,g", "Generate example neural net config file")
      ("samples,s", po::value<std::string>())
      ("config,c", po::value<std::string>(), "Network configuration file of key=value pairs")
      ("labels,l", po::value<std::vector<std::string>>()->multitoken(),
        "List of class labels, e.g. cat dog")
      ("network,n", po::value<std::string>()->required(), "File to save/load neural network state");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    optionChoice(vm, { "train", "eval", "gen" });
    optionDependency(vm, "train", "samples");
    optionDependency(vm, "train", "labels");
    optionDependency(vm, "train", "config");
    optionDependency(vm, "eval", "samples");
    conflictingOptions(vm, "eval", "labels");
    conflictingOptions(vm, "eval", "config");
    conflictingOptions(vm, "gen", "samples");
    conflictingOptions(vm, "gen", "config");
    conflictingOptions(vm, "gen", "labels");
    conflictingOptions(vm, "gen", "network");

    if (vm.count("gen")) {
      NetworkConfig::printExample(std::cout);
      return 0;
    }

    const bool trainingMode = vm.count("train");
    const std::string networkFile = vm["network"].as<std::string>();
    const std::string samplesFile = vm["samples"].as<std::string>();

    po::notify(vm);

    const auto t1 = high_resolution_clock::now();

    if (trainingMode) {
      std::cout << "Training classifier" << std::endl;

      std::vector<std::string> classes = vm["labels"].as<std::vector<std::string>>();
      std::string configFile = vm["config"].as<std::string>();

      NetworkConfig config = NetworkConfig::fromFile(configFile);

      Classifier classifier(config, classes);

      TrainingData trainingData(loadCsvData(samplesFile, classifier.inputSize(), classes));
      trainingData.normalize();

      classifier.train(trainingData);

      classifier.toFile(networkFile);
    }
    else {
      Classifier classifier(networkFile);

      std::cout << "Testing classifier" << std::endl;

      TestData testData(loadCsvData(samplesFile, classifier.inputSize(), classifier.classLabels()));
      testData.normalize(classifier.trainingSetMin(), classifier.trainingSetMax());
      Classifier::Results results = classifier.test(testData);

      std::cout << "Correct classifications: "
        << results.good << "/" << results.good + results.bad << std::endl;

      std::cout << "Average cost: " << results.cost << std::endl;
    }

    const auto t2 = high_resolution_clock::now();
    const long long elapsed = duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Running time: " << elapsed << " milliseconds" << std::endl;
  }
  catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
