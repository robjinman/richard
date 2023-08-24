#include <iostream>
#include <random>

int main() {
  std::default_random_engine generator;
  std::normal_distribution<double> clusterAX(0.3, 0.1);
  std::normal_distribution<double> clusterAY(0.4, 0.03);
  std::normal_distribution<double> clusterBX(0.7, 0.1);
  std::normal_distribution<double> clusterBY(0.8, 0.05);

  std::cout << "A,B" << std::endl;

  size_t N = 100;

  for (int i = 0; i < N; ++i) {
    double aX = clusterAX(generator);
    double aY = clusterAY(generator);

    std::cout << "A," << aX << "," << aY << std::endl;

    double bX = clusterBX(generator);
    double bY = clusterBY(generator);

    std::cout << "B," << aX << "," << aY << std::endl;
  }

  return 0;
}
