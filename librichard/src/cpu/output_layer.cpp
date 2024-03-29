#include "richard/cpu/output_layer.hpp"
#include "richard/utils.hpp"
#include "richard/config.hpp"

namespace richard {
namespace cpu {

OutputLayer::OutputLayer(const Config& config, size_t inputSize) {
  initialize(config, inputSize);

  m_W.randomize(0.1f);
}

OutputLayer::OutputLayer(const Config& config, std::istream& stream, size_t inputSize) {
  initialize(config, inputSize);

  stream.read(reinterpret_cast<char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void OutputLayer::initialize(const Config& config, size_t inputSize) {
  m_activationFn = sigmoid;
  m_activationFnPrime = sigmoidPrime;

  size_t size = config.getNumber<size_t>("size");
  m_learnRate = config.getNumber<netfloat_t>("learnRate");
  m_learnRateDecay = config.getNumber<netfloat_t>("learnRateDecay");

  m_B = Vector(size);
  m_W = Matrix(inputSize, size);

  m_inputDelta = Vector(inputSize);
  m_deltaB = Vector(size);
  m_deltaW = Matrix(inputSize, size);
}

void OutputLayer::writeToStream(std::ostream& stream) const {
  stream.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.write(reinterpret_cast<const char*>(m_W.data()),
    m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

const DataArray& OutputLayer::activations() const {
  return m_A.storage();
}

const DataArray& OutputLayer::inputDelta() const {
  return m_inputDelta.storage();
}

DataArray OutputLayer::evalForward(const DataArray& inputs) const {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  Vector y = (m_W * x + m_B).computeTransform(m_activationFn);

  return y.storage();
}

Size3 OutputLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

void OutputLayer::trainForward(const DataArray& inputs) {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  m_Z = m_W * x + m_B;
  m_A = m_Z.computeTransform(m_activationFn);
}

void OutputLayer::updateDeltas(const DataArray& inputs, const DataArray& outputs) {
  ConstVectorPtr pY = Vector::createShallow(outputs);
  const Vector& y = *pY;

  Vector deltaC = quadraticCostDerivatives(m_A, y);
  Vector delta = m_Z.computeTransform(m_activationFnPrime).hadamard(deltaC);

  m_inputDelta = m_W.transposeMultiply(delta);

  m_deltaW += outerProduct(delta, inputs);
  m_deltaB += delta;
}

void OutputLayer::updateParams(size_t epoch) {
  netfloat_t learnRate = m_learnRate * static_cast<netfloat_t>(pow(m_learnRateDecay, epoch));

  m_W -= m_deltaW * learnRate;
  m_B -= m_deltaB * learnRate;

  m_deltaB.zero();
  m_deltaW.zero();
}

void OutputLayer::test_setWeights(const DataArray& W) {
  m_W = Matrix(W, m_W.cols(), m_W.rows());
}

void OutputLayer::test_setBiases(const DataArray& B) {
  m_B = B;
}

const Matrix& OutputLayer::test_deltaW() const {
  return m_deltaW;
}

const Vector& OutputLayer::test_deltaB() const {
  return m_deltaB;
}

const Matrix& OutputLayer::test_W() const {
  return m_W;
}

const Vector& OutputLayer::test_B() const {
  return m_B;
}

void OutputLayer::test_setActivationFn(ActivationFn f, ActivationFn fPrime) {
  m_activationFn = f;
  m_activationFnPrime = fPrime;
}

}
}
