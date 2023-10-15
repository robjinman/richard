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

Vector foo(const Vector& A, const Vector& B) {
  auto doubler = [](double x) -> double {
    return x * 2.0;
  };

  Vector V = A + B.transform(doubler);
  return V;
}

TEST_F(VectorTest, assignmentOperatorInScope) {
  Vector a({0, 0, 0});

  {
    Vector b{3, 3, 3};
    Vector c({7, 7, 7});
    a = foo(b, c);
  }

  ASSERT_EQ(Vector({17, 17, 17}), a);
}

TEST_F(VectorTest, dotProduct) {
  Vector a{1, 2, 3};
  Vector b{4, 5, 6};

  ASSERT_EQ(32, a.dot(b));
}

TEST_F(VectorTest, elementPlusEquals) {
  Vector a{1, 2, 3};
  a[1] += 11;

  ASSERT_EQ(a[1], 13);
}
