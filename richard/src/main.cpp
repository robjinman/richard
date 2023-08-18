#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/program_options.hpp>
#include "neural_net.hpp"

const std::string DESCRIPTION = "Richard is gaining power";

namespace po = boost::program_options;

std::vector<TrainingSample> loadTrainingData(const std::string& filePath) {
  std::ifstream fin(filePath);
  std::vector<TrainingSample> trainingData;

  const size_t dimensions = 2;
  char label = '0';
  Vector sample{dimensions};

  for (std::string line; std::getline(fin, line); ) {
    std::stringstream ss{line};
    for (size_t i = 0; ss.good(); ++i) {
      std::string token;
      std::getline(ss, token, ',');

      if (i == 0 && token.length() > 0) {
        label = token[0];
      }
      else {
        sample[i - 1] = std::stod(token);
      }
    }

    trainingData.push_back(TrainingSample{label, std::move(sample)});
  }

  return trainingData;
}

void trainNetwork() {
  NeuralNet net{2, { 4, 4 }, 2};
  const std::string filePath = "data/train.csv";

  std::vector<TrainingSample> trainingData = loadTrainingData(filePath);

  for (const TrainingSample& sample : trainingData) {
    net.train(sample);
  }

  // TODO: Persist network weights to file
}

void testNetwork() {
  NeuralNet net{2, { 4, 4 }, 2};

  // TODO: Load network weights from file

  std::cout << net.evaluate({ 34.0, 23.0 });
  std::cout << net.evaluate({ 12.0, 60.0 });
  std::cout << net.evaluate({ 24.0, 11.0 });
}

int main(int argc, char** argv) {
  try {
    po::options_description desc{DESCRIPTION};
    desc.add_options()
      ("help,h", "Show help")
      ("train,t", po::bool_switch());

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    po::notify(vm);

    if (vm["train"].as<bool>()) {
      std::cout << "Training neural net" << std::endl;
      trainNetwork();
    }
    else {
      std::cout << "Evaluating neural net" << std::endl;
      testNetwork();
    }
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
