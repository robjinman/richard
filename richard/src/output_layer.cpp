#include <iostream> // TODO
#include "output_layer.hpp"

OutputLayer::OutputLayer(const nlohmann::json& obj, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1) {

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
  , m_delta(1) {

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

const Vector& OutputLayer::activations() const {
  return m_A;
}

const Vector& OutputLayer::delta() const {
  return m_delta;
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

Vector OutputLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

std::array<size_t, 3> OutputLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

void OutputLayer::trainForward(const Vector& inputs) {
  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);
}

void OutputLayer::updateDelta(const Vector& layerInputs, const Vector& y, size_t epoch) {
  Vector deltaC = quadraticCostDerivatives(m_A, y);
  m_delta = m_Z.transform(sigmoidPrime).hadamard(deltaC);

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * learnRate;
}
