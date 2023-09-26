#include "output_layer.hpp"

OutputLayer::OutputLayer(size_t numNeurons, size_t inputSize, double learnRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate) {

  m_B = Vector(numNeurons);
  m_B.randomize(1.0);

  m_W = Matrix(inputSize, numNeurons);
  m_W.randomize(1.0);
}

OutputLayer::OutputLayer(std::istream& fin, size_t numNeurons, size_t inputSize, double learnRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate) {

  m_B = Vector(numNeurons);
  fin.read(reinterpret_cast<char*>(m_B.data()), numNeurons * sizeof(double));

  m_W = Matrix(inputSize, numNeurons);
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
  return nlohmann::json();
}

Vector OutputLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

std::array<size_t, 2> OutputLayer::outputSize() const {
  return { m_B.size(), 1 };
}

void OutputLayer::trainForward(const Vector& inputs) {
  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);
}

void OutputLayer::updateDelta(const Vector& layerInputs, const Vector& y) {
  Vector deltaC = quadraticCostDerivatives(m_A, y);
  m_delta = m_Z.transform(sigmoidPrime).hadamard(deltaC);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * m_learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * m_learnRate;
}
