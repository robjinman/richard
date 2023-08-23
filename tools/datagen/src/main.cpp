#include <iostream>
#include <random>

int main() {
  std::default_random_engine generator;
  std::normal_distribution<double> maleHeight(1.75, 0.1);
  std::normal_distribution<double> maleWeight(80.1, 11.2);
  std::normal_distribution<double> femaleHeight(1.56, 0.08);
  std::normal_distribution<double> femaleWeight(67.5, 9.3);

  std::cout << "M,F" << std::endl;

  size_t N = 10000;

  for (int i = 0; i < N; ++i) {
    double manH = maleHeight(generator);
    double manW = maleWeight(generator);

    std::cout << "M," << manH << "," << manW << std::endl;

    double womanH = femaleHeight(generator);
    double womanW = femaleWeight(generator);

    std::cout << "F," << womanH << "," << womanW << std::endl;
  }

  return 0;
}
