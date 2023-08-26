#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <map>
#include <chrono>
#include <boost/program_options.hpp>
#include "neural_net.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace po = boost::program_options;

namespace {

const std::string DESCRIPTION = "Richard is gaining power";

// Load training data from csv file
//
// The first line is the set of possible labels, which should be a single character.
// Subsequent lines are a label followed by float data values.
// E.g.
//
// a,b,c
// b,23.1,45.5
// a,44.0,52.1
// c,11.9,92.4
// ...
std::unique_ptr<Dataset> loadData(const std::string& filePath, size_t inputSize) {
  std::ifstream fin(filePath);

  std::string line;
  std::vector<char> classLabels;

  std::getline(fin, line);

  std::stringstream ss{line};
  while (ss.good()) {
    std::string token;
    std::getline(ss, token, ',');
    classLabels.push_back(token[0]);
  }

  auto data = std::make_unique<Dataset>(classLabels);

  while (std::getline(fin, line)) {
    std::stringstream ss{line};
    char label = '0';
    Vector sample(inputSize);

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

    //sample.normalize();
    data->addSample(label, sample);
  }

  return data;
}

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

    const std::string networkWeightsFile = "../data/ocr/network_weights"; // TODO
    const std::string trainingDataFile = "../data/ocr/train.csv";
    const std::string testDataFile = "../data/ocr/test.csv";

    po::notify(vm);

    const auto t1 = high_resolution_clock::now();

    NeuralNet net{784, 300, 80, 10};

    if (vm["train"].as<bool>()) {
      std::cout << "Training neural net" << std::endl;

      TrainingData trainingData(loadData(trainingDataFile, net.inputSize()));
      trainingData.normalize();

      net.train(trainingData);

      net.toFile(trainingData, networkWeightsFile);
    }
    else {
      Vector trainingDataMin(1);
      Vector trainingDataMax(1);
      net.fromFile(networkWeightsFile, trainingDataMin, trainingDataMax);

      std::cout << "Evaluating neural net" << std::endl;

      TestData testData(loadData(testDataFile, net.inputSize()));
      testData.normalize(trainingDataMin, trainingDataMax);
      NeuralNet::Results results = net.test(testData);

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
