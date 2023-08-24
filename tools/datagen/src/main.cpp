#include <iostream>
#include <random>

int main() {
  std::default_random_engine generator;
  std::normal_distribution<double> clusterAX(-0.7, 0.2);
  std::normal_distribution<double> clusterAY(0.6, 0.06);
  std::normal_distribution<double> clusterBX(0.68, 0.2);
  std::normal_distribution<double> clusterBY(-0.4, 0.1);

  std::cout << "A,B" << std::endl;

  size_t N = 100;

  for (int i = 0; i < N; ++i) {
    double aX = clusterAX(generator);
    double aY = clusterAY(generator);

    std::cout << "A," << aX << "," << aY << std::endl;

    double bX = clusterBX(generator);
    double bY = clusterBY(generator);

    std::cout << "B," << bX << "," << bY << std::endl;
  }

  return 0;
}
