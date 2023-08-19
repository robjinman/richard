#include <gtest/gtest.h>
#include <math.hpp>

class VectorTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(VectorTest, equalityOperator) {
  Vector a{1, 2, 3, 4, 5, 4, 3, 2, 1};
  Vector b{1, 2, 3, 4};
  Vector c{1, 2, 3, 3, 5, 4, 3, 2, 1};
  Vector d{1, 2, 3, 4, 5, 4, 3, 2, 1};

  ASSERT_NE(a, b);
  ASSERT_NE(a, c);
  ASSERT_EQ(a, d);
}

TEST_F(VectorTest, copyContructor) {
  Vector a{1, 2, 3};
  Vector b(a);

  ASSERT_EQ(Vector({1, 2, 3}), b);
}

TEST_F(VectorTest, moveContructor) {
  Vector a{1, 2, 3};
  Vector b(std::move(a));

  ASSERT_EQ(Vector({1, 2, 3}), b);
}

TEST_F(VectorTest, assignmentOperator) {
  Vector a{1, 2, 3};
  Vector b = a;

  ASSERT_EQ(Vector({1, 2, 3}), b);
}

TEST_F(VectorTest, dotProduct) {
  Vector a{1, 2, 3};
  Vector b{4, 5, 6};

  ASSERT_EQ(32, a.dot(b));
}
