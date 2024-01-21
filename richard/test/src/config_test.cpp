#include <config.hpp>
#include <gtest/gtest.h>
#include <vector>

using namespace richard;

class ConfigTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(ConfigTest, fromJsonString) {
  const std::string json = R"(
    {
      "object": {
        "number": 123,
        "array": [ 1, 2, 3, 4 ]
      },
      "number": 234
    }
  )";

  Config config = Config::fromJson(json);
  EXPECT_EQ(config.getValue<long>("number"), 234);
}

TEST_F(ConfigTest, getArray) {
  const std::string json = R"(
    {
      "array": [ 3, 4, 5, 6, 7 ]
    }
  )";

  Config config = Config::fromJson(json);

  auto arr = config.getArray<long>("array");
  std::vector<long> expected{ 3, 4, 5, 6, 7 };

  EXPECT_EQ(arr, expected);
}

TEST_F(ConfigTest, setNumber) {
  Config config;
  config.setValue("number", 678);

  EXPECT_EQ(config.getValue<long>("number"), 678);
}

TEST_F(ConfigTest, setNumberCoercedType) {
  Config config;
  config.setValue("number", 678u);

  EXPECT_EQ(config.getValue<long>("number"), 678);
}

TEST_F(ConfigTest, getArrayCoercedType) {
  const std::string json = R"(
    {
      "array": [ 3, 4, 5, 6, 7 ]
    }
  )";

  Config config = Config::fromJson(json);

  auto arr = config.getArray<int>("array");
  std::vector<int> expected{ 3, 4, 5, 6, 7 };

  EXPECT_EQ(arr, expected);
}

TEST_F(ConfigTest, getFloatArrayAsInt) {
  const std::string json = R"(
    {
      "array": [ 3.1, 4.7, 5.6, 6.3, 7.2 ]
    }
  )";

  Config config = Config::fromJson(json);

  auto arr = config.getArray<int>("array");
  std::vector<int> expected{ 3, 4, 5, 6, 7 };

  EXPECT_EQ(arr, expected);
}

TEST_F(ConfigTest, setArray) {
  Config config;

  std::vector<long> arr{ 3, 4, 5, 6 };
  config.setArray("array", arr);

  EXPECT_EQ(config.getArray<long>("array"), arr);
}

TEST_F(ConfigTest, setArrayCoercedType) {
  const std::string json = R"(
    {
      "number": 123
    }
  )";

  Config config = Config::fromJson(json);

  std::vector<int> arr{ 3, 4, 5, 6 };
  config.setArray("array", arr);

  EXPECT_EQ(config.getArray<int>("array"), arr);
}

TEST_F(ConfigTest, getStdArray) {
  const std::string json = R"(
    {
      "array": [ 3, 4, 5, 6, 7 ]
    }
  )";

  Config config = Config::fromJson(json);

  auto arr = config.getArray<long, 5>("array");
  std::array<long, 5> expected{3, 4, 5, 6, 7};

  EXPECT_EQ(arr, expected);
}

TEST_F(ConfigTest, getStdArrayCoercedType) {
  const std::string json = R"(
    {
      "array": [ 3, 4, 5, 6, 7 ]
    }
  )";

  Config config = Config::fromJson(json);

  auto arr = config.getArray<size_t, 5>("array");
  std::array<size_t, 5> expected{3, 4, 5, 6, 7};

  EXPECT_EQ(arr, expected);
}

TEST_F(ConfigTest, getObject) {
  const std::string json = R"(
    {
      "object": {
        "number": 123,
        "array": [ 1, 2, 3, 4 ]
      },
      "number": 234
    }
  )";

  Config config = Config::fromJson(json);
  Config object = config.getObject("object");

  EXPECT_EQ(object.getValue<long>("number"), 123);
}

TEST_F(ConfigTest, getObjectArray) {
  const std::string json = R"(
    {
      "array": [
        {
          "number": 123
        },
        {
          "number": 234
        },
        {
          "number": 345
        }
      ]
    }
  )";

  Config config = Config::fromJson(json);
  std::vector<Config> arr = config.getObjectArray("array");

  std::array<long, 3> expected{ 123, 234, 345 };

  size_t i = 0;
  for (auto obj : arr) {
    EXPECT_EQ(obj.getValue<long>("number"), expected[i]);
    ++i;
  }
}

TEST_F(ConfigTest, dump) {
  const std::string json = R"(
    {
      "object": {
        "number": 123,
        "array": [ 1, 2, 3, 4 ]
      },
      "number": 234,
      "objects": [
        {
          "foo": "hello",
          "bar": 45.6
        },
        {
          "foo": "world",
          "bar": 56.23
        }
      ]
    }
  )";

  Config config1 = Config::fromJson(json);

  std::string json2 = config1.dump();

  Config config2 = Config::fromJson(json2);

  EXPECT_EQ(config1, config2);
}
