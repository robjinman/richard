#include "exception.hpp"
#include "utils.hpp"
#include "classifier_training_app.hpp"
#include "classifier_eval_app.hpp"
#include "file_system.hpp"
#include "logger.hpp"
#include <boost/program_options.hpp>
#include <chrono>

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

ApplicationPtr constructApp(Logger& logger, FileSystem& fileSystem, po::variables_map& vm) {
  ApplicationPtr app = nullptr;

  if (vm.count("train")) {
    ClassifierTrainingApp::Options opts;

    vm.erase("train");

    opts.samplesPath = getOpt(vm, "samples", true).as<std::string>();
    opts.configFile = getOpt(vm, "config", true).as<std::string>();
    opts.networkFile = getOpt(vm, "network", true).as<std::string>();
    opts.gpuAccelerated = vm.count("gpu");

    vm.erase("gpu");

    app = std::make_unique<ClassifierTrainingApp>(fileSystem, opts, logger);
  }
  else if (vm.count("eval")) {
    ClassifierEvalApp::Options opts;

    vm.erase("eval");

    opts.samplesPath = getOpt(vm, "samples", true).as<std::string>();
    opts.networkFile = getOpt(vm, "network", true).as<std::string>();
    opts.gpuAccelerated = vm.count("gpu");

    vm.erase("gpu");

    app = std::make_unique<ClassifierEvalApp>(fileSystem, opts, logger);
  }
  else {
    EXCEPTION("Missing required argument: train or eval");
  }

  for (auto i : vm) {
    logger.warn(STR("Unused option '" << i.first << "'"));
  }

  return app;
}

void printExampleConfig(Logger& logger, const std::string& appType) {
  nlohmann::json obj;

  if (appType == "train") {
    obj = ClassifierTrainingApp::exampleConfig();
  }
  else {
    EXCEPTION("Expected app type to be one of ['train'], got '" << appType << "'");
  }

  logger.info(obj.dump(4));
}

}

int main(int argc, char** argv) {
  LoggerPtr logger = createStdoutLogger();

  try {
    po::options_description desc{"Richard is gaining power"};
    desc.add_options()
      ("help,h", "Show help")
      ("train,t", "Train a classifier")
      ("eval,e", "Evaluate a classifier with test data")
      ("gen,g", po::value<std::string>(), "Generate example config file for app type [train]")
      ("samples,s", po::value<std::string>())
      ("config,c", po::value<std::string>(), "JSON configuration file")
      ("network,n", po::value<std::string>()->required(), "File to save/load neural network state")
      ("gpu,x", "Use GPU acceleration");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") || argc == 1) {
      logger->info(STR(desc));
      return EXIT_SUCCESS;
    }

    optionChoice(vm, { "train", "eval", "gen" });

    if (vm.count("gen")) {
      printExampleConfig(*logger, getOpt(vm, "gen", true).as<std::string>());

      for (auto i : vm) {
        logger->warn(STR("Unused option '" << i.first << "'"));
      }

      return EXIT_SUCCESS;
    }

    FileSystemPtr fileSystem = createFileSystem();
    ApplicationPtr app = constructApp(*logger, *fileSystem, vm);

    auto t1 = std::chrono::high_resolution_clock::now();

    app->start();

    auto t2 = std::chrono::high_resolution_clock::now();
    long long elapsed = duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    logger->info(STR("Running time: " << elapsed << " milliseconds"));
  }
  catch (const std::exception& e) {
    logger->error(e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

