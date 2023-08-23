#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <map>
#include <boost/program_options.hpp>
#include "neural_net.hpp"

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
TrainingData loadTrainingData(const std::string& filePath) {
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

  TrainingData trainingData(classLabels);

  std::streampos dataStart = fin.tellg();
  std::getline(fin, line);
  size_t dimensions = std::count(line.begin(), line.end(), ',');
  fin.seekg(dataStart);

  while (std::getline(fin, line)) {
    std::stringstream ss{line};
    char label = '0';
    Vector sample(dimensions);

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

    trainingData.addSample(label, sample);
  }

  return trainingData;
}

void trainNetwork(NeuralNet& net) {
  //NeuralNet net{2, { 4, 4 }, 2};
  const std::string filePath = "../data/bmi/train.csv"; // TODO

  TrainingData trainingData = loadTrainingData(filePath);
  net.train(trainingData);

  // TODO: Persist network weights to file
}

bool outputsMatch(const Vector& x, const Vector& y) {
  auto largestComponent = [](const Vector& v) {
    double largest = std::numeric_limits<double>::min();
    size_t largestIdx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i] > largest) {
        largest = v[i];
        largestIdx = i;
      }
    }
    return largestIdx;
  };

  return largestComponent(x) == largestComponent(y);
}

void testNetwork(const NeuralNet& net) {
  //NeuralNet net{2, { 4, 4 }, 2};
  const std::string filePath = "../data/bmi/test.csv"; // TODO

  TrainingData testData = loadTrainingData(filePath);

  size_t good = 0;
  size_t bad = 0;
  for (const auto& sample : testData.data()) {
    Vector actual = net.evaluate(sample.data);
    Vector expected = testData.classOutputVector(sample.label);

    //std::cout << sample.data;
    //std::cout << actual;

    if (outputsMatch(actual, expected)) {
      ++good;
    }
    else {
      ++bad;
    }
  }

  std::cout << "Correct classifications = " << good << "/" << good + bad << std::endl;
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

    po::notify(vm);

    NeuralNet net{2, { 4, 4 }, 2};

    if (vm["train"].as<bool>()) {
      std::cout << "Training neural net" << std::endl;
      trainNetwork(net);
    }
    else {
      std::cout << "Training neural net" << std::endl;
      trainNetwork(net);

      std::cout << "Evaluating neural net" << std::endl;
      testNetwork(net);
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
