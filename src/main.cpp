#include <iostream>
#include "neural_net.hpp"

int main() {
  NeuralNet net{2, { 4, 4 }, 2};

  std::cout << net.evaluate({ 34.0, 23.0 });
  std::cout << net.evaluate({ 12.0, 60.0 });
  std::cout << net.evaluate({ 24.0, 11.0 });

  return 0;
}
