#include <richard/csv_data_loader.hpp>
#include <gtest/gtest.h>

using namespace richard;

class CsvDataLoaderTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(CsvDataLoaderTest, loadSamples) {
  std::unique_ptr<std::istream> ss = std::make_unique<std::stringstream>("1,0,255,128");

  NormalizationParams normalization;
  normalization.min = 0;
  normalization.max = 255;

  CsvDataLoader loader(std::move(ss), 3, normalization, 1000);
  std::vector<Sample> samples = loader.loadSamples();

  ASSERT_EQ(samples.size(), 1);

  VectorPtr pX = Vector::createShallow(samples[0].data.storage());

  ASSERT_EQ(*pX, Vector({ 0.f, 1.f, 128.f / 255.f }));
}

