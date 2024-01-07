#include "cpu/output_layer.hpp"
#include "utils.hpp"

namespace richard {
namespace cpu {

OutputLayer::OutputLayer(const nlohmann::json& obj, size_t inputSize) {
  initialize(obj, inputSize);

  m_W.randomize(0.1);
}

OutputLayer::OutputLayer(const nlohmann::json& obj, std::istream& stream, size_t inputSize) {
  initialize(obj, inputSize);

  stream.read(reinterpret_cast<char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void OutputLayer::initialize(const nlohmann::json& obj, size_t inputSize) {
  m_activationFn = sigmoid;
  m_activationFnPrime = sigmoidPrime;

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();

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

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      m_deltaW.set(k, j, m_deltaW.at(k, j) + inputs[k] * delta[j]);
    }
  }

  m_deltaB += delta;
}

void OutputLayer::updateParams(size_t epoch) {
  netfloat_t learnRate = m_learnRate * pow(m_learnRateDecay, epoch);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      m_W.set(k, j, m_W.at(k, j) - m_deltaW.at(k, j) * learnRate);
    }
  }

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
