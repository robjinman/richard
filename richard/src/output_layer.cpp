#include <iostream>
#include "output_layer.hpp"

OutputLayer::OutputLayer(const nlohmann::json& obj, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();

  m_B = Vector(size);

  m_W = Matrix(inputSize, size);
  m_W.randomize(0.1);
}

OutputLayer::OutputLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_activationFn(sigmoid)
  , m_activationFnPrime(sigmoidPrime) {

  size_t size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();

  m_B = Vector(size);
  fin.read(reinterpret_cast<char*>(m_B.data()), size * sizeof(double));

  m_W = Matrix(inputSize, size);
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

void OutputLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
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

nlohmann::json OutputLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "output";
  config["size"] = m_B.size();
  config["learnRate"] = m_learnRate;
  config["learnRateDecay"] = m_learnRateDecay;
  return config;
}

DataArray OutputLayer::evalForward(const DataArray& inputs) const {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  Vector y = (m_W * x + m_B).computeTransform(m_activationFn);

  return y.storage();
}

std::array<size_t, 3> OutputLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

void OutputLayer::trainForward(const DataArray& inputs) {
  ConstVectorPtr pX = Vector::createShallow(inputs);
  const Vector& x = *pX;

  m_Z = m_W * x + m_B;
  m_A = m_Z.computeTransform(m_activationFn);
}

void OutputLayer::updateDelta(const DataArray& inputs, const DataArray& outputs, size_t epoch) {
  ConstVectorPtr pY = Vector::createShallow(outputs);
  const Vector& y = *pY;

  Vector deltaC = quadraticCostDerivatives(m_A, y);
  //std::cout << "deltaC:\n";
  //std::cout << deltaC;

  m_delta = m_Z.computeTransform(m_activationFnPrime).hadamard(deltaC);

  //std::cout << "Output layer delta:\n";
  //std::cout << m_delta;

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = inputs[k] * m_delta[j] * learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * learnRate;
}

void OutputLayer::setWeights(const Matrix& W) {
  m_W = W;
}

void OutputLayer::setBiases(const Vector& B) {
  m_B = B;
}

void OutputLayer::setWeights(const std::vector<DataArray>& W) {
  ASSERT(W.size() == 1);

  m_W = Matrix(W[0], m_W.cols(), m_W.rows());
}

void OutputLayer::setBiases(const DataArray& B) {
  m_B = B;
}

void OutputLayer::setActivationFn(ActivationFn f, ActivationFn fPrime) {
  m_activationFn = f;
  m_activationFnPrime = fPrime;
}
