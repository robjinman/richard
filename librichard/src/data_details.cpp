#include "richard/data_details.hpp"

namespace richard {

NormalizationParams::NormalizationParams()
  : min(0)
  , max(0) {}

NormalizationParams::NormalizationParams(const Config& config)
  : min(config.getNumber<netfloat_t>("min"))
  , max(config.getNumber<netfloat_t>("max")) {}

const Config& NormalizationParams::exampleConfig() {
  static Config config = []() {
    Config c;
    c.setNumber("min", 0);
    c.setNumber("max", 255);
    return c;
  }();

  return config;
}

DataDetails::DataDetails(const Config& config)
  : normalization(config.getObject("normalization"))
  , classLabels(config.getStringArray("classes"))
  , shape(config.getNumberArray<size_t, 3>("shape")) {}

const Config& DataDetails::exampleConfig() {
  static Config config = []() {
    Config c;
    c.setObject("normalization", NormalizationParams::exampleConfig());
    c.setStringArray("classes", {
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    });
    c.setNumberArray<size_t>("shape", { 28, 28, 1 });
    return c;
  }();

  return config;
}

}
