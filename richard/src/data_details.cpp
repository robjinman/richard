#include "data_details.hpp"

namespace richard {

NormalizationParams::NormalizationParams()
  : min(0)
  , max(0) {}

NormalizationParams::NormalizationParams(const Config& config)
  : min(config.getValue<netfloat_t>("min"))
  , max(config.getValue<netfloat_t>("max")) {}

const Config& NormalizationParams::exampleConfig() {
  static Config obj;
  static bool done = false;

  if (!done) {
    obj.setValue("min", 0);
    obj.setValue("max", 255);

    done = true;
  }

  return obj;
}

DataDetails::DataDetails(const Config& config)
  : normalization(config.getObject("normalization"))
  , classLabels(config.getArray<std::string>("classes"))
  , shape(config.getArray<size_t, 3>("shape")) {}

const Config& DataDetails::exampleConfig() {
  static Config obj;
  static bool done = false;

  if (!done) {
    obj.setObject("normalization", NormalizationParams::exampleConfig());
    obj.setArray<std::string>("classes", {
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    });
    obj.setArray<size_t>("shape", { 28, 28, 1 });

    done = true;
  }

  return obj;
}

}
