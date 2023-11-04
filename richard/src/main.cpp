#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include "exception.hpp"
#include "classifier_training_app.hpp"
#include "classifier_eval_app.hpp"

namespace po = boost::program_options;

using std::chrono::duration_cast;

namespace {

const std::string DESCRIPTION = "Richard is gaining power";

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

po::variable_value getOpt(po::variables_map& vm, const std::string& option, bool required) {
  if (required && vm.count(option) == 0) {
    EXCEPTION("Missing argument '" << option << "'");
  }

  auto value = vm.at(option);
  vm.erase(option);

  return value;
};

ApplicationPtr constructApp(po::variables_map& vm) {
  ApplicationPtr app = nullptr;

  if (vm.count("train")) {
    ClassifierTrainingApp::Options opts;

    vm.erase("train");

    opts.samplesPath = getOpt(vm, "samples", true).as<std::string>();
    opts.configFile = getOpt(vm, "config", true).as<std::string>();
    opts.networkFile = getOpt(vm, "network", true).as<std::string>();

    app = std::make_unique<ClassifierTrainingApp>(opts);
  }
  else if (vm.count("eval")) {
    ClassifierEvalApp::Options opts;

    vm.erase("eval");

    opts.samplesPath = getOpt(vm, "samples", true).as<std::string>();
    opts.networkFile = getOpt(vm, "network", true).as<std::string>();

    app = std::make_unique<ClassifierEvalApp>(opts);
  }
  else {
    EXCEPTION("Missing required argument: train or eval");
  }

  for (auto i : vm) {
    std::cerr << "Warning: Unused option '" << i.first << "'" << std::endl;
  }

  return app;
}

}

// richard --train --samples ../../data/ocr/train.csv --config ../../data/ocr/config.json --network ../../data/ocr/network
// richard --eval --samples ../../data/ocr/test.csv --network ../../data/ocr/network

// richard --train --samples ../../data/catdog/train --config ../../data/catdog/config.json --network ../../data/catdog/network
// richard --eval --samples ../../data/catdog/test --network ../../data/catdog/network
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

    if (vm.count("gen")) {
      // TODO
      //nlohmann::json obj;
      //obj["classifier"] = Classifier::defaultConfig();
      //std::cout << obj.dump(4) << std::endl;
      return EXIT_SUCCESS;
    }

    ApplicationPtr app = constructApp(vm);

    const auto t1 = std::chrono::high_resolution_clock::now();

    app->start();
    
    const auto t2 = std::chrono::high_resolution_clock::now();
    const long long elapsed = duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Running time: " << elapsed << " milliseconds" << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

