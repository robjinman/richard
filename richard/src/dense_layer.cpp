#include <iostream> // TODO
#include "dense_layer.hpp"
#include "util.hpp"

DenseLayer::DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<double>();

  m_B = Vector(numNeurons);
  fin.read(reinterpret_cast<char*>(m_B.data()), numNeurons * sizeof(double));

  m_W = Matrix(inputSize, numNeurons);
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

DenseLayer::DenseLayer(const nlohmann::json& obj, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<double>();

  m_B = Vector(numNeurons);
  m_W = Matrix(inputSize, numNeurons);

  m_W.randomize(0.1);
}

void DenseLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

std::array<size_t, 3> DenseLayer::outputSize() const {
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

nlohmann::json DenseLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "dense";
  config["size"] = m_B.size();
  config["learnRate"] = m_learnRate;
  config["learnRateDecay"] = m_learnRateDecay;
  config["dropoutRate"] = m_dropoutRate;
  return config;
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

void DenseLayer::updateDelta(const DataArray& inputs, const Layer& nextLayer, size_t epoch) {
  ConstVectorPtr pNextDelta = Vector::createShallow(nextLayer.delta());

  m_delta = nextLayer.W().transposeMultiply(*pNextDelta)
                         .hadamard(m_Z.computeTransform(m_activationFnPrime));

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = inputs[k] * m_delta[j] * learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * learnRate;

  //std::cout << "Dense layer delta:\n";
  //std::cout << m_delta;
}

void DenseLayer::setWeights(const Matrix& W) {
  m_W = W;
}

void DenseLayer::setBiases(const Vector& B) {
  m_B = B;
}

void DenseLayer::setWeights(const std::vector<DataArray>& W) {
  ASSERT(W.size() == 1);

  m_W = Matrix(W[0], m_W.cols(), m_W.rows());
}

void DenseLayer::setBiases(const DataArray& B) {
  m_B = B;
}

void DenseLayer::setActivationFn(ActivationFn f, ActivationFn fPrime) {
  m_activationFn = f;
  m_activationFnPrime = fPrime;
}
