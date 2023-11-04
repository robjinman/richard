#include <gtest/gtest.h>
#include <csv_data_loader.hpp>

class CsvDataLoaderTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CsvDataLoaderTest, loadSamples) {
  std::stringstream ss;
  ss << "1,0,255,128";
  std::vector<Sample> samples;

  NormalizationParams normalization;
  normalization.min = 0;
  normalization.max = 255;

  CsvDataLoader loader("Dummy", 3, normalization);
  loader.loadSamples(ss, samples, 1);

  ASSERT_EQ(samples.size(), 1);

  VectorPtr pX = Vector::createShallow(samples[0].data.storage());

  ASSERT_EQ(*pX, Vector({ 0, 1.0, 128.0 / 255.0 }));
}

