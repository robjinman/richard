#include <math.hpp>
#include <gtest/gtest.h>

using namespace richard;

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

TEST_F(MathTest, arrayCopyConstructor) {
  Array a({ 1, 2, 3 });
  Array b(a);
  
  ASSERT_EQ(b, Array({ 1, 2, 3 }));
}

TEST_F(MathTest, arrayCopyConstructorRhsShallow) {
  Array a({ 1, 2, 3, 4, 5 });
  ConstArrayPtr b = a.subvector(1, 3);

  Array c(*b);

  ASSERT_EQ(c, Array({ 2, 3, 4 }));
}

TEST_F(MathTest, arrayMoveConstuctor) {
  Array a({ 1, 2, 3 });
  Array b(std::move(a));

  ASSERT_EQ(b, Array({ 1, 2, 3 }));
  ASSERT_EQ(a.size(), 0);
}

TEST_F(MathTest, arrayMoveConstructorRhsShallow) {
  Array a({ 1, 2, 3, 4, 5 });
  ArrayPtr pB = a.subvector(1, 3);
  Array& b = *pB;

  // b isn't moved because it's shallow
  Array c(std::move(b));

  ASSERT_EQ(c, Array({ 2, 3, 4 }));
  ASSERT_EQ(b.size(), 3); // Didn't actually move b
}

TEST_F(MathTest, arrayAssignment) {
  Array a(1);

  {
    Array b({ 1, 2, 3, 4 });
    a = b;
  }

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
}

TEST_F(MathTest, arrayAssignmentRhsShallow) {
  Array b({ 1, 2, 3, 4 });
  ConstArrayPtr c = b.subvector(0, 4);

  Array a(1);
  a = *c;

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
}

TEST_F(MathTest, arrayAssignmentRhsShallowMismatchedSizes) {
  // TODO
}

TEST_F(MathTest, arrayAssignmentRhsShallowRValue) {
  Array b({ 1, 2, 3, 4 });
  ArrayPtr c = b.subvector(0, 4);

  Array a(1);
  // When RHS is shallow, this should perform a copy, not a move, even when RHS is an r-value
  a = std::move(*c);

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));

  // c is still valid and unchanged
  ASSERT_EQ(*c, Array({ 1, 2, 3, 4 }));

  // b is still valid and unchanged
  ASSERT_EQ(b, Array({ 1, 2, 3, 4 }));
}

TEST_F(MathTest, arrayAssignmentLhsShallow) {
  Array a({ 1, 2, 3, 4, 5 });
  ArrayPtr pB = a.subvector(1, 3);
  Array& b = *pB;

  Array c({ 9, 9, 9 });
  b = c;

  ASSERT_EQ(a, Vector({ 1, 9, 9, 9, 5 }));
  ASSERT_EQ(b, Vector({ 9, 9, 9 }));
  ASSERT_EQ(c, Vector({ 9, 9, 9 }));
}

TEST_F(MathTest, arrayAssignmentLhsShallowMismatchedSizes) {
  // TODO
}

TEST_F(MathTest, arrayAssignmentLhsShallowRhsRValue) {
  Array a({ 1, 2, 3, 4, 5 });
  ArrayPtr pB = a.subvector(1, 3);
  Array& b = *pB;

  Array c({ 9, 9, 9 });
  // Not actually moved because LHS is shallow
  b = std::move(c);

  ASSERT_EQ(a, Vector({ 1, 9, 9, 9, 5 }));
  ASSERT_EQ(b, Vector({ 9, 9, 9 }));
  ASSERT_EQ(c, Vector({ 9, 9, 9 })); // c wasn't actually moved
}

TEST_F(MathTest, arrayAssignmentRhsRValue) {
  Array a(1);

  {
    Array b({ 1, 2, 3, 4 });
    a = std::move(b);

    // b's data has been moved, so b is now empty
    ASSERT_EQ(b.size(), 0);
  }

  ASSERT_EQ(a, Array({ 1, 2, 3, 4 }));
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

TEST_F(MathTest, array2CopyConstructor) {
  Array2 a({
    { 1, 2 },
    { 3, 4 }
  });

  Array2 b(a);

  ASSERT_EQ(b, Array2({
    { 1, 2 },
    { 3, 4 }
  }));
}

TEST_F(MathTest, array2CopyConstructorRhsShallow) {
  Vector a({ 1, 2, 3, 4 });
  Array2Ptr b = Array2::createShallow(a.storage(), 2, 2);

  Array2 c(*b);

  ASSERT_EQ(c, Array2({
    { 1, 2 },
    { 3, 4 }
  }));
}

TEST_F(MathTest, array2MoveConstuctor) {
  Array2 a({
    { 1, 2 },
    { 3, 4 }
  });

  Array2 b(std::move(a));

  ASSERT_EQ(b, Array2({
    { 1, 2 },
    { 3, 4 }
  }));

  ASSERT_EQ(a.size(), 0);
}

TEST_F(MathTest, array2MoveConstructorRhsShallow) {
  Vector a({ 1, 2, 3, 4 });
  Array2Ptr pB = Array2::createShallow(a.storage(), 2, 2);
  Array2& b = *pB;

  Array2 c(std::move(b));

  ASSERT_EQ(c, Array2({
    { 1, 2 },
    { 3, 4 }
  }));
  
  ASSERT_EQ(b, Array2({ // Didn't actually move b
    { 1, 2 },
    { 3, 4 }
  }));
}

TEST_F(MathTest, array2Assignment) {
  Array2 a(1, 1);

  {
    Array2 b({
      { 1, 2 },
      { 3, 4 }
    });
    a = b;
  }

  ASSERT_EQ(a, Array2({
    { 1, 2 },
    { 3, 4 }
  }));
}

TEST_F(MathTest, array2AssignmentRhsShallow) {
  Array3 b({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  });
  ConstArray2Ptr c = b.slice(1);

  Array2 a(1, 1);
  a = *c;

  ASSERT_EQ(a, Array2({
    { 5, 6 },
    { 7, 8 }
  }));
}

TEST_F(MathTest, array2AssignmentRhsShallowMismatchedSizes) {
  // TODO
}

TEST_F(MathTest, array2AssignmentRhsShallowRValue) {
  Array3 b({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  });
  ConstArray2Ptr c = b.slice(1);

  Array2 a(1, 1);
  // When RHS is shallow, this should perform a copy, not a move, even when RHS is an r-value
  a = std::move(*c);

  ASSERT_EQ(a, Array2({
    { 5, 6 },
    { 7, 8 }
  }));

  // c is still valid and unchanged
  ASSERT_EQ(*c, Array2({
    { 5, 6 },
    { 7, 8 }
  }));

  // b is still valid and unchanged
  ASSERT_EQ(b, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array2AssignmentLhsShallow) {
  Array3 a({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  });
  Array2Ptr pB = a.slice(1);
  Array2& b = *pB;

  Array2 c({
    { 9, 9 },
    { 9, 9 }
  });
  b = c;

  ASSERT_EQ(a, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 9, 9 },
      { 9, 9 }
    }
  }));
  ASSERT_EQ(b, Array2({
    { 9, 9 },
    { 9, 9 }
  }));
  ASSERT_EQ(c, Array2({
    { 9, 9 },
    { 9, 9 }
  }));
}

TEST_F(MathTest, array2AssignmentLhsShallowMismatchedSizes) {
  // TODO
}

TEST_F(MathTest, array2AssignmentLhsShallowRhsRValue) {
  Array3 a({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  });
  Array2Ptr pB = a.slice(1);
  Array2& b = *pB;

  Array2 c({
    { 9, 9 },
    { 9, 9 }
  });
  // Not actually moved because LHS is shallow
  b = std::move(c);

  ASSERT_EQ(a, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 9, 9 },
      { 9, 9 }
    }
  }));
  ASSERT_EQ(b, Array2({ // Wasn't actually moved
    { 9, 9 },
    { 9, 9 }
  }));
  ASSERT_EQ(c, Array2({
    { 9, 9 },
    { 9, 9 }
  }));
}

TEST_F(MathTest, array2AssignmentRhsRValue) {
  Array2 a(1, 1);

  {
    Array2 b({
      { 1, 2 },
      { 3, 4 }
    });
    a = std::move(b);

    // b's data has been moved, so b is now empty
    ASSERT_EQ(b.size(), 0);
  }

  ASSERT_EQ(a, Array2({
      { 1, 2 },
      { 3, 4 }
    }));
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

TEST_F(MathTest, array3CopyConstructor) {
  Array3 a({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  });

  Array3 b(a);

  ASSERT_EQ(b, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array3CopyConstructorRhsShallow) {
  Vector a({ 1, 2, 3, 4, 5, 6, 7, 8 });
  Array3Ptr b = Array3::createShallow(a.storage(), 2, 2, 2);

  Array3 c(*b);

  ASSERT_EQ(c, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array3MoveConstuctor) {
  Array3 a({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  });

  Array3 b(std::move(a));

  ASSERT_EQ(b, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));

  ASSERT_EQ(a.size(), 0);
}

TEST_F(MathTest, array3MoveConstructorRhsShallow) {
  Vector a({ 1, 2, 3, 4, 5, 6, 7, 8 });
  Array3Ptr pB = Array3::createShallow(a.storage(), 2, 2, 2);
  Array3& b = *pB;

  Array3 c(std::move(b));

  ASSERT_EQ(c, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
  
  ASSERT_EQ(b, Array3({ // Didn't actually move b
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array3Assignment) {
  Array3 a(1, 1, 1);

  {
    Array3 b({
      {
        { 1, 2 },
        { 3, 4 }
      },
      {
        { 5, 6 },
        { 7, 8 }
      }
    });
    a = b;
  }

  ASSERT_EQ(a, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array3AssignmentRhsShallow) {
  Vector v({ 1, 2, 3, 4, 5, 6, 7, 8 });  
  
  ConstArray3Ptr c = Array3::createShallow(v.storage(), 2, 2, 2);

  Array3 a(1, 1, 1);
  a = *c;

  ASSERT_EQ(a, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array3AssignmentRhsShallowMismatchedSizes) {
  // TODO
}

TEST_F(MathTest, array3AssignmentRhsShallowRValue) {
  Vector b({ 1, 2, 3, 4, 5, 6, 7, 8 });
  ConstArray3Ptr c = Array3::createShallow(b.storage(), 2, 2, 2);

  Array3 a(1, 1, 1);
  // When RHS is shallow, this should perform a copy, not a move, even when RHS is an r-value
  a = std::move(*c);

  ASSERT_EQ(a, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));

  // c is still valid and unchanged
  ASSERT_EQ(*c, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));

  // b is still valid and unchanged
  ASSERT_EQ(b, Vector({ 1, 2, 3, 4, 5, 6, 7, 8 }));
}

TEST_F(MathTest, array3AssignmentLhsShallow) {
  Vector a({ 1, 2, 3, 4, 5, 6, 7, 8 });
  Array3Ptr pB = Array3::createShallow(a.storage(), 2, 2, 2);
  Array3& b = *pB;

  Array3 c({
    {
      { 9, 9 },
      { 9, 9 }
    },
    {
      { 9, 9 },
      { 9, 9 }
    }
  });
  b = c;

  ASSERT_EQ(a, Vector({ 9, 9, 9, 9, 9, 9, 9, 9 }));
  ASSERT_EQ(b, c);
}

TEST_F(MathTest, array3AssignmentLhsShallowMismatchedSizes) {
  // TODO
}

TEST_F(MathTest, array3AssignmentLhsShallowRhsRValue) {
  Vector a({ 1, 2, 3, 4, 5, 6, 7, 8 });
  Array3Ptr pB = Array3::createShallow(a.storage(), 2, 2, 2);
  Array3& b = *pB;

  Array3 c({
    {
      { 9, 9 },
      { 9, 9 }
    },
    {
      { 9, 9 },
      { 9, 9 }
    }
  });
  // Not actually moved because LHS is shallow
  b = std::move(c);

  ASSERT_EQ(a, Vector({ 9, 9, 9, 9, 9, 9, 9, 9 }));
  ASSERT_EQ(b, Array3({
    {
      { 9, 9 },
      { 9, 9 }
    },
    {
      { 9, 9 },
      { 9, 9 }
    }
  }));
  ASSERT_EQ(c, Array3({ // Wasn't actually moved
    {
      { 9, 9 },
      { 9, 9 }
    },
    {
      { 9, 9 },
      { 9, 9 }
    }
  }));
}

TEST_F(MathTest, array3AssignmentRhsRValue) {
  Array3 a(1, 1, 1);

  {
    Array3 b({
      {
        { 1, 2 },
        { 3, 4 }
      },
      {
        { 5, 6 },
        { 7, 8 }
      }
    });
    a = std::move(b);
    
    ASSERT_EQ(b.size(), 0);
  }

  ASSERT_EQ(a, Array3({
    {
      { 1, 2 },
      { 3, 4 }
    },
    {
      { 5, 6 },
      { 7, 8 }
    }
  }));
}

TEST_F(MathTest, array3ElementAccess) {
  Array3 arr3({
    {
      { 3, 4 },
      { 7, 2 },
      { 9, 1 }
    },
    {
      { 1, 0 },
      { 6, 9 },
      { 4, 8 }
    }
  });

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

TEST_F(MathTest, computeCrossCorrelation) {
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

  computeCrossCorrelation(image, kernel, featureMap);

  ASSERT_EQ(featureMap, Array2({
    { 6+45+6+10+30+0+2+0, 9+5+15+12+54+0+0+2 },
    { 2+25+21+16+6+0+6+7, 5+30+24+4+0+0+14+4 },
    { 7+40+12+12+18+0+12+2, 8+10+18+14+42+0+4+3 }
  }));
}

TEST_F(MathTest, convolutionIsCrossCorrelationWithReversedKernel) {
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

  Kernel kernel1({{
    { 1, 5 },
    { 3, 2 }
  }, {
    { 6, 0 },
    { 2, 1 }
  }});

  size_t kW = kernel1.W();
  size_t kH = kernel1.H();
  size_t kD = kernel1.D();

  Kernel kernel2(kW, kH, kD);
  for (size_t k = 0; k < kD; ++k) {
    for (size_t j = 0; j < kH; ++j) {
      for (size_t i = 0; i < kW; ++i) {
        kernel2.set(i, j, k, kernel1.at(kW - i - 1, kH - j - 1, k));
      }
    }
  }

  Array2 convResult(2, 3);
  Array2 xCorrResult(2, 3);

  computeCrossCorrelation(image, kernel1, xCorrResult);
  computeConvolution(image, kernel2, convResult);

  ASSERT_EQ(convResult, xCorrResult);
}

TEST_F(MathTest, computeFullCrossCorrelation) {
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

  Array2 featureMap(4, 5);

  computeFullCrossCorrelation(image, kernel, featureMap);

  ASSERT_EQ(featureMap, Array2({
    { 12+5, 18+18+10+9, 27+2+18+1, 3+2 },
    { 30+4+0+1, 6+45+6+10+30+0+2+0, 9+5+15+12+54+0+0+2, 1+18+6+4 },
    { 10+14+0+3, 2+25+21+16+6+0+6+7, 5+30+24+4+0+0+14+4, 6+6+12+8 },
    { 35+8+0+6, 7+40+12+12+18+0+12+2, 8+10+18+14+42+0+4+3, 2+21+24+6 },
    { 20+0, 4+30+36+0, 6+35+12+0, 7+18 }
  }));
}

TEST_F(MathTest, fullConvolutionIsFullCrossCorrelationWithReversedKernel) {
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

  Kernel kernel1({{
    { 1, 5 },
    { 3, 2 }
  }, {
    { 6, 0 },
    { 2, 1 }
  }});

  size_t kW = kernel1.W();
  size_t kH = kernel1.H();
  size_t kD = kernel1.D();

  Kernel kernel2(kW, kH, kD);
  for (size_t k = 0; k < kD; ++k) {
    for (size_t j = 0; j < kH; ++j) {
      for (size_t i = 0; i < kW; ++i) {
        kernel2.set(i, j, k, kernel1.at(kW - i - 1, kH - j - 1, k));
      }
    }
  }

  Array2 convResult(4, 5);
  Array2 xCorrResult(4, 5);

  computeFullCrossCorrelation(image, kernel1, xCorrResult);
  computeFullConvolution(image, kernel2, convResult);

  ASSERT_EQ(convResult, xCorrResult);
}
