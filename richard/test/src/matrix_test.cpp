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

TEST_F(MatrixTest, constructor_initializationList) {
  Matrix m({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  });

  ASSERT_EQ(1, m.at(0, 0));
  ASSERT_EQ(2, m.at(1, 0));
  ASSERT_EQ(3, m.at(2, 0));
  ASSERT_EQ(4, m.at(0, 1));
  ASSERT_EQ(5, m.at(1, 1));
  ASSERT_EQ(6, m.at(2, 1));
  ASSERT_EQ(7, m.at(0, 2));
  ASSERT_EQ(8, m.at(1, 2));
  ASSERT_EQ(9, m.at(2, 2));
}
