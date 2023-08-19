#include <gtest/gtest.h>
#include <math.hpp>

class MathTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(MathTest, dotProduct) {
  Vector a{1, 2, 3};
  Vector b{4, 5, 6};

  ASSERT_EQ(32, a.dot(b));
}
