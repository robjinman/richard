#include "neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"
#include "logger.hpp"
#include <atomic>

namespace {

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  DBG_ASSERT(actual.size() == expected.size());
  return (expected - actual).squareMagnitude() * 0.5;
};

class GpuNeuralNet : public NeuralNet {
  public:
    using CostFn = std::function<netfloat_t(const Vector&, const Vector&)>;

    GpuNeuralNet(const Triple& inputShape, const nlohmann::json& config, Logger& logger);
    GpuNeuralNet(const Triple& inputShape, const nlohmann::json& config, std::istream& s,
      Logger& logger);

    CostFn costFn() const override;
    Triple inputSize() const override;
    void writeToStream(std::ostream& s) const override;
    void train(LabelledDataSet& data) override;
    VectorPtr evaluate(const Array3& inputs) const override;

    void abort() override;

  private:
    Logger& m_logger;
    bool m_isTrained;
    Triple m_inputShape;
    std::atomic<bool> m_abort;
};

void GpuNeuralNet::abort() {
  m_abort = true;
}

GpuNeuralNet::GpuNeuralNet(const Triple& inputShape, const nlohmann::json& config, Logger& logger)
  : m_logger(logger)
  , m_isTrained(false)
  , m_inputShape(inputShape) {

}

GpuNeuralNet::GpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger)
  : m_logger(logger)
  , m_isTrained(false)
  , m_inputShape(inputShape) {

}

NeuralNet::CostFn GpuNeuralNet::costFn() const {
  return quadradicCost;
}

void GpuNeuralNet::writeToStream(std::ostream& fout) const {
  ASSERT_MSG(m_isTrained, "Neural net is not trained");

}

Triple GpuNeuralNet::inputSize() const {
  return m_inputShape;
}

void GpuNeuralNet::train(LabelledDataSet& trainingData) {

}

VectorPtr GpuNeuralNet::evaluate(const Array3& x) const {

}

}

NeuralNetPtr createGpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  Logger& logger) {

  return std::make_unique<GpuNeuralNet>(inputShape, config, logger);
}

NeuralNetPtr createGpuNeuralNet(const Triple& inputShape, const nlohmann::json& config,
  std::istream& fin, Logger& logger) {

  return std::make_unique<GpuNeuralNet>(inputShape, config, fin, logger);
}
