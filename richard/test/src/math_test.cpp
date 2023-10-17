#include <gtest/gtest.h>
#include <math.hpp>

class MathTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(MathTest, arrayEquality) {
  Array a({ 1, 2, 3, 4, 5, 4, 3, 2, 1 });
  Array b({ 1, 2, 3, 4 });
  Array c({ 1, 2, 3, 3, 5, 4, 3, 2, 1 });
  Array d({ 1, 2, 3, 4, 5, 4, 3, 2, 1 });

  ASSERT_NE(a, b);
  ASSERT_NE(a, c);
  ASSERT_EQ(a, d);
}

TEST_F(MathTest, arrayCopyAssignment) {
  Array a(1);

  {
    Array b({ 1, 2, 3, 4 });
    a = b;
  }

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
}

TEST_F(MathTest, arrayCopyAssignmentShallow) {
  Array b({ 1, 2, 3, 4 });
  ArrayPtr c = b.subvector(0, 4);

  Array a(1);
  a = *c;

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
}

TEST_F(MathTest, arrayMoveAssignment) {
  Array a(1);

  {
    Array b({ 1, 2, 3, 4 });
    a = std::move(b);

    ASSERT_EQ(b.size(), 0);
  }

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
}

TEST_F(MathTest, arrayMoveAssignmentShallow) {
  Array a(1);
  Array b({ 1, 2, 3, 4 });

  {
    ArrayPtr c = b.subvector(0, 4);
    a = std::move(*c);

    ASSERT_EQ(c->size(), 0);
  }

  a[2] = 9;

  ASSERT_EQ(a, Array({ 1, 2, 9, 4 }));
  ASSERT_EQ(b, Array({ 1, 2, 9, 4 }));
}

TEST_F(MathTest, shallowArrayAssignment) {
  Array a({ 1, 2, 3, 4 });
  ArrayPtr b = a.subvector(0, 4);

  // After this, b is no longer a shallow vector pointing to a
  *b = Vector({ 5, 6, 7, 8 });

  (*b)[2] = 100;

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
  ASSERT_EQ(*b, Array({ 5, 6, 100, 8 }));
}

TEST_F(MathTest, vectorElementPlusEquals) {
  Vector a{ 1, 2, 3 };
  a[1] += 11;

  ASSERT_EQ(a[1], 13);
}

TEST_F(MathTest, vectorDotProduct) {
  Vector a{1, 2, 3};
  Vector b{4, 5, 6};

  ASSERT_EQ(32, a.dot(b));
}

TEST_F(MathTest, array2Equality) {
  Array2 a({
    { 1, 2, 3 },
    { 4, 5, 6 }
  });

  Array2 b({
    { 1, 2 },
    { 3, 4 },
    { 5, 6 }
  });

  Array2 c({
    { 1, 2, 3 },
    { 4, 4, 6 }
  });

  Array2 d({
    { 1, 2, 3 },
    { 4, 5, 6 }
  });

  ASSERT_NE(a, b);
  ASSERT_NE(a, c);
  ASSERT_EQ(a, d);
}

TEST_F(MathTest, array2ElementAccess) {
  Array2 arr2({
    { 3, 4 },
    { 7, 2 },
    { 9, 1 }
  });

  ASSERT_EQ(arr2.at(0, 0), 3);
  ASSERT_EQ(arr2.at(1, 0), 4);
  ASSERT_EQ(arr2.at(1, 2), 1);
}

TEST_F(MathTest, array3ElementAccess) {
  Array3 arr3({{
    { 3, 4 },
    { 7, 2 },
    { 9, 1 }
  }, {
    { 1, 0 },
    { 6, 9 },
    { 4, 8 }
  }});

  ASSERT_EQ(arr3.at(0, 0, 0), 3);
  ASSERT_EQ(arr3.at(1, 0, 0), 4);

  ASSERT_EQ(arr3.at(1, 2, 0), 1);
  ASSERT_EQ(arr3.at(1, 1, 1), 9);
}

TEST_F(MathTest, array2AsArray) {
  auto foo = [](const DataArray& data) {
    ConstArrayPtr arrPtr = Array::createShallow(data);
    const Array& arr = *arrPtr;

    ASSERT_EQ(arr, Array({ 1, 2, 3, 4, 5, 6 }));
  };

  Array2 arr2({
    { 1, 2, 3 },
    { 4, 5, 6 }
  });

  foo(arr2.storage());
}

TEST_F(MathTest, arrayAsArray2) {
  auto foo = [](const DataArray& data, size_t w, size_t h) {
    ConstArray2Ptr arr2Ptr = Array2::createShallow(data, w, h);
    const Array2& arr2 = *arr2Ptr;

    ASSERT_EQ(arr2, Array2({
      { 1, 2, 3 },
      { 4, 5, 6 }
    }));
  };

  Array2 arr({
    { 1, 2, 3, 4, 5, 6 }
  });

  foo(arr.storage(), 3, 2);
}

TEST_F(MathTest, arrayEqualsItselfPlus) {
  Vector v({ 1, 2, 3, 4 });

  v = v + 3;

  ASSERT_EQ(v, Vector({ 4, 5, 6, 7 }));
}

TEST_F(MathTest, vectorMinusEqualsVectorMultiply) {
  Vector a{ 1, 2, 3 };
  Vector b({ 3, 4, 5 });
  a = a - b * 2.0;

  ASSERT_EQ(a, Vector({ -5, -6, -7 }));
}

TEST_F(MathTest, constSliceArray2) {
  const Array2 arr2({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  });

  ConstArrayPtr slicePtr = arr2.slice(1);
  const Array& slice = *slicePtr;

  ASSERT_EQ(slice, Array({ 4, 5, 6 }));
}

TEST_F(MathTest, sliceArray2Modify) {
  Array2 arr2({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  });

  ArrayPtr slicePtr = arr2.slice(1);
  Array& slice = *slicePtr;

  slice[0] = 11;
  slice[1] = 12;
  slice[2] = 13;

  ASSERT_EQ(arr2, Array2({
    { 1, 2, 3 },
    { 11, 12, 13 },
    { 7, 8, 9 }
  }));
}

TEST_F(MathTest, convolve) {
  Array3 image({{
    { 6, 9, 1 },
    { 2, 5, 6 },
    { 7, 8, 2 },
    { 4, 6, 7 }
  }, {
    { 5, 9, 1 },
    { 1, 0, 2 },
    { 3, 7, 4 },
    { 6, 2, 3 },
  }});

  Kernel kernel({{
    { 1, 5 },
    { 3, 2 }
  }, {
    { 6, 0 },
    { 2, 1 }
  }});

  Array2 featureMap(2, 3);

  kernel.convolve(image, featureMap);

  ASSERT_EQ(featureMap, Array2({
    { 6+45+6+10+30+0+2+0, 9+5+15+12+54+0+0+2 },
    { 2+25+21+16+6+0+6+7, 5+30+24+4+0+0+14+4 },
    { 7+40+12+12+18+0+12+2, 8+10+18+14+42+0+4+3 }
  }));
}
