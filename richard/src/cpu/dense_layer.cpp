#include "cpu/dense_layer.hpp"
#include "utils.hpp"
#include "config.hpp"

namespace richard {
namespace cpu {

DenseLayer::DenseLayer(const Config& config, size_t inputSize) {
  initialize(config, inputSize);

  m_W.randomize(0.1);
}

DenseLayer::DenseLayer(const Config& config, std::istream& stream, size_t inputSize) {
  initialize(config, inputSize);

  stream.read(reinterpret_cast<char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void DenseLayer::initialize(const Config& config, size_t inputSize) {
  m_activationFn = sigmoid;
  m_activationFnPrime = sigmoidPrime;

  size_t size = config.getInteger("size");
  m_learnRate = config.getFloat("learnRate");
  m_learnRateDecay = config.getFloat("learnRateDecay");
  m_dropoutRate = config.getFloat("dropoutRate");

  m_B = Vector(size);
  m_W = Matrix(inputSize, size);

  m_inputDelta = Vector(inputSize);
  m_deltaB = Vector(size);
  m_deltaW = Matrix(inputSize, size);
}

void DenseLayer::writeToStream(std::ostream& stream) const {
  stream.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.write(reinterpret_cast<const char*>(m_W.data()),
    m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

Size3 DenseLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

const DataArray& DenseLayer::activations() const {
  return m_A.storage();
}

const DataArray& DenseLayer::inputDelta() const {
  return m_inputDelta.storage();
}

DataArray DenseLayer::evalForward(const DataArray& inputs) const {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  Vector y = (m_W * x + m_B).computeTransform(m_activationFn);

  return y.storage();
}

void DenseLayer::trainForward(const DataArray& inputs) {
  auto shouldDrop = [this]() {
    return rand() / (RAND_MAX + 1.0) < m_dropoutRate;
  };

  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  m_Z = m_W * x + m_B;
  m_A = m_Z.computeTransform(m_activationFn);

  for (size_t a = 0; a < m_A.size(); ++a) {
    if (shouldDrop()) {
      m_A[a] = 0.0;
    }
  }
}

void DenseLayer::updateDeltas(const DataArray& inputs, const DataArray& outputDelta) {
  ConstVectorPtr pDeltaA = Vector::createShallow(outputDelta);
  const Vector& deltaA = *pDeltaA;

  Vector delta = deltaA.hadamard(m_Z.computeTransform(m_activationFnPrime));
  m_inputDelta = m_W.transposeMultiply(delta);

  m_deltaW += outerProduct(delta, inputs);
  m_deltaB += delta;
}

void DenseLayer::updateParams(size_t epoch) {
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

void DenseLayer::test_setWeights(const DataArray& W) {
  m_W = Matrix(W, m_W.cols(), m_W.rows());
}

void DenseLayer::test_setBiases(const DataArray& B) {
  m_B = B;
}

const Matrix& DenseLayer::test_deltaW() const {
  return m_deltaW;
}

const Vector& DenseLayer::test_deltaB() const {
  return m_deltaB;
}

const Matrix& DenseLayer::test_W() const {
  return m_W;
}

const Vector& DenseLayer::test_B() const {
  return m_B;
}

void DenseLayer::test_setActivationFn(ActivationFn f, ActivationFn fPrime) {
  m_activationFn = f;
  m_activationFnPrime = fPrime;
}

}
}
