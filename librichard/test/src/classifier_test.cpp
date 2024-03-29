#include "mock_file_system.hpp"
#include "mock_logger.hpp"
#include "mock_platform_paths.hpp"
#include <richard/config.hpp>
#include <richard/classifier.hpp>
#include <richard/data_details.hpp>
#include <richard/event_system.hpp>
#include <gtest/gtest.h>

using namespace richard;
using testing::NiceMock;

class ClassifierTest : public testing::Test {
  public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(ClassifierTest, exampleConfig) {
  auto eventSystem = createEventSystem();
  auto platformPaths = createPlatformPaths();
  auto fileSystem = createFileSystem();
  NiceMock<MockLogger> logger;

  Config config = Classifier::exampleConfig();
  DataDetails dataDetails{DataDetails::exampleConfig()};
  Classifier classifier{dataDetails, config, *eventSystem, *fileSystem, *platformPaths, logger,
    true};
}
