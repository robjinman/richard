#include "cpu/dense_layer.hpp"
#include "util.hpp"

DenseLayer::DenseLayer(const nlohmann::json& obj, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_deltaB(1)
  , m_deltaW(1, 1)
  , m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<netfloat_t>();

  m_B = Vector(size);
  m_W = Matrix(inputSize, size);

  m_W.randomize(0.1);

  m_delta = Vector(size);
  m_deltaB = Vector(size);
  m_deltaW = Matrix(inputSize, size);
}

DenseLayer::DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_deltaB(1)
  , m_deltaW(1, 1)
  , m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<netfloat_t>();

  m_B = Vector(size);
  fin.read(reinterpret_cast<char*>(m_B.data()), size * sizeof(netfloat_t));

  m_W = Matrix(inputSize, size);
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));

  m_delta = Vector(size);
  m_deltaB = Vector(size);
  m_deltaW = Matrix(inputSize, size);
}

void DenseLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  fout.write(reinterpret_cast<const char*>(m_W.data()),
    m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

Triple DenseLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

const DataArray& DenseLayer::activations() const {
  return m_A.storage();
}

const DataArray& DenseLayer::delta() const {
  return m_delta.storage();
}

const Matrix& DenseLayer::W() const {
  return m_W;
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

void DenseLayer::updateDelta(const DataArray& inputs, const Layer& nextLayer) {
  ConstVectorPtr pNextDelta = Vector::createShallow(nextLayer.delta());

  m_delta = nextLayer.W().transposeMultiply(*pNextDelta)
                         .hadamard(m_Z.computeTransform(m_activationFnPrime));

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      m_deltaW.set(k, j, m_deltaW.at(k, j) + inputs[k] * m_delta[j]);
    }
  }

  m_deltaB += m_delta;
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

void DenseLayer::setWeights(const Matrix& W) {
  m_W = W;
}

void DenseLayer::setBiases(const Vector& B) {
  m_B = B;
}

void DenseLayer::setWeights(const DataArray& W) {
  m_W = Matrix(W, m_W.cols(), m_W.rows());
}

void DenseLayer::setBiases(const DataArray& B) {
  m_B = B;
}

void DenseLayer::setActivationFn(ActivationFn f, ActivationFn fPrime) {
  m_activationFn = f;
  m_activationFnPrime = fPrime;
}
