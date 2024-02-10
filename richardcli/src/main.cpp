#include "outputter.hpp"
#include "classifier_training_app.hpp"
#include "classifier_eval_app.hpp"
#include <richard/exception.hpp>
#include <richard/utils.hpp>
#include <richard/file_system.hpp>
#include <richard/platform_paths.hpp>
#include <richard/event_system.hpp>
#include <richard/logger.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>

namespace po = boost::program_options;

using namespace richard;
using std::chrono::duration_cast;

namespace {

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

ApplicationPtr constructApp(EventSystem& eventSystem, Outputter& outputter, Logger& logger,
  FileSystem& fileSystem, const PlatformPaths& platformPaths, po::variables_map& vm) {

  ApplicationPtr app = nullptr;

  if (vm.count("train")) {
    ClassifierTrainingApp::Options opts;

    vm.erase("train");

    opts.samplesPath = getOpt(vm, "samples", true).as<std::string>();
    opts.configFile = getOpt(vm, "config", true).as<std::string>();
    opts.networkFile = getOpt(vm, "network", true).as<std::string>();
    opts.gpuAccelerated = vm.count("gpu");

    vm.erase("gpu");

    app = std::make_unique<ClassifierTrainingApp>(eventSystem, fileSystem, platformPaths, opts,
      outputter, logger);
  }
  else if (vm.count("eval")) {
    ClassifierEvalApp::Options opts;

    vm.erase("eval");

    opts.samplesPath = getOpt(vm, "samples", true).as<std::string>();
    opts.networkFile = getOpt(vm, "network", true).as<std::string>();
    opts.gpuAccelerated = vm.count("gpu");

    vm.erase("gpu");

    app = std::make_unique<ClassifierEvalApp>(eventSystem, fileSystem, platformPaths, opts,
      outputter, logger);
  }
  else {
    EXCEPTION("Missing required argument: train or eval");
  }

  for (auto i : vm) {
    logger.warn(STR("Unused option '" << i.first << "'"));
  }

  return app;
}

void printExampleConfig(Outputter& outputter, const std::string& appType) {
  Config config;

  if (appType == "train") {
    config = ClassifierTrainingApp::exampleConfig();
  }
  else {
    EXCEPTION("Expected app type to be one of ['train'], got '" << appType << "'");
  }

  outputter.printLine(config.dump(4));
}

void printHeader(Outputter& outputter, const std::string& appName, bool gpuAccelerated) {
  outputter.printBanner();
  outputter.printLine(STR("[ Mode: " << appName << " ]"));
  outputter.printLine(STR("[ GPU accelaration: " << (gpuAccelerated ? "ON" : "OFF") << " ]"));
  outputter.printSeparator();
}

po::variables_map parseProgramArgs(po::options_description& desc, int argc, const char** argv) {
    desc.add_options()
      ("help,h", "Show help")
      ("train,t", "Train a classifier")
      ("eval,e", "Evaluate a classifier with test data")
      ("gen,g", po::value<std::string>(), "Generate example config file for app type [train]")
      ("samples,s", po::value<std::string>(), "Path to data samples")
      ("config,c", po::value<std::string>(), "JSON configuration file")
      ("network,n", po::value<std::string>()->required(), "File to save/load neural network state")
      ("log,l", po::value<std::string>(), "Log file path")
      ("gpu,x", "Use GPU acceleration");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    return vm;
}

}

int main(int argc, const char** argv) {
  Outputter outputter{std::cout};
  std::ofstream logStream;
  LoggerPtr logger = nullptr;

  try {
    po::options_description desc{"Richard is gaining power"};
    auto vm = parseProgramArgs(desc, argc, argv);

    if (vm.count("help") || argc == 1) {
      outputter.printLine(STR(desc));
      return EXIT_SUCCESS;
    }

    optionChoice(vm, { "train", "eval", "gen" });

    if (vm.count("log")) {
      logStream = std::ofstream{vm.at("log").as<std::string>()};
      logger = createLogger(logStream, logStream, logStream, logStream);
      vm.erase("log");
    }
    else {
      logger = createLogger(std::cerr, std::cerr, std::cout, std::cout);
    }

    if (vm.count("gen")) {
      printExampleConfig(outputter, getOpt(vm, "gen", true).as<std::string>());

      for (auto i : vm) {
        logger->warn(STR("Unused option '" << i.first << "'"));
      }

      return EXIT_SUCCESS;
    }

    bool gpuAccelerated = vm.count("gpu");

    FileSystemPtr fileSystem = createFileSystem();
    PlatformPathsPtr platformPaths = createPlatformPaths();
    EventSystemPtr eventSystem = createEventSystem();
    ApplicationPtr app = constructApp(*eventSystem, outputter, *logger, *fileSystem, *platformPaths,
      vm);

    printHeader(outputter, app->name(), gpuAccelerated);

    auto t1 = std::chrono::high_resolution_clock::now();

    app->start();

    auto t2 = std::chrono::high_resolution_clock::now();
    long long elapsed = duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    outputter.printSeparator();
    outputter.printLine(STR("Running time: " << elapsed << " milliseconds"));
  }
  catch (const std::exception& e) {
    if (logger != nullptr) {
      logger->error(e.what());
    }
    else {
      std::cerr << e.what() << std::endl;
    }
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
