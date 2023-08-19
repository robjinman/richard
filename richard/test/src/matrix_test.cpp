#include <gtest/gtest.h>
#include <math.hpp>

class MatrixTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(MatrixTest, zero) {
  size_t cols = 4;
  size_t rows = 3;
  Matrix m{cols, rows};

  m.zero();

  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      ASSERT_EQ(0, m.at(c, r));
    }
  }
}
