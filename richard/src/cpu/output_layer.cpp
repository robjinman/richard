#include "cpu/output_layer.hpp"
#include "util.hpp"

namespace richard {
namespace cpu {

OutputLayer::OutputLayer(const nlohmann::json& obj, size_t inputSize)
  : m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();

  m_B = Vector(size);

  m_W = Matrix(inputSize, size);
  m_W.randomize(0.1);
  
  m_delta = Vector(size);
  m_deltaB = Vector(size);
  m_deltaW = Matrix(inputSize, size);
}

OutputLayer::OutputLayer(const nlohmann::json& obj, std::istream& stream, size_t inputSize)
  : m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();

  m_B = Vector(size);
  stream.read(reinterpret_cast<char*>(m_B.data()), size * sizeof(netfloat_t));

  m_W = Matrix(inputSize, size);
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));

  m_delta = Vector(size);
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

const DataArray& OutputLayer::delta() const {
  return m_delta.storage();
}

const Matrix& OutputLayer::W() const {
  return m_W;
}

DataArray OutputLayer::evalForward(const DataArray& inputs) const {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  Vector y = (m_W * x + m_B).computeTransform(m_activationFn);

  return y.storage();
}

Triple OutputLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

void OutputLayer::trainForward(const DataArray& inputs) {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  m_Z = m_W * x + m_B;
  m_A = m_Z.computeTransform(m_activationFn);
}

void OutputLayer::updateDelta(const DataArray&, const Layer&) {
  EXCEPTION("Use other OutputLayer::updateDelta() overload");
}

void OutputLayer::updateDelta(const DataArray& inputs, const DataArray& outputs) {
  ConstVectorPtr pY = Vector::createShallow(outputs);
  const Vector& y = *pY;

  Vector deltaC = quadraticCostDerivatives(m_A, y);
  m_delta = m_Z.computeTransform(m_activationFnPrime).hadamard(deltaC);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      m_deltaW.set(k, j, m_deltaW.at(k, j) + inputs[k] * m_delta[j]);
    }
  }

  m_deltaB += m_delta;
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

void OutputLayer::setWeights(const Matrix& W) {
  m_W = W;
}

void OutputLayer::setBiases(const Vector& B) {
  m_B = B;
}

void OutputLayer::setWeights(const DataArray& W) {
  m_W = Matrix(W, m_W.cols(), m_W.rows());
}

void OutputLayer::setBiases(const DataArray& B) {
  m_B = B;
}

void OutputLayer::setActivationFn(ActivationFn f, ActivationFn fPrime) {
  m_activationFn = f;
  m_activationFnPrime = fPrime;
}

}
}
