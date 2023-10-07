#include "dense_layer.hpp"

DenseLayer::DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1) {

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
  , m_delta(1) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<double>();

  m_B = Vector(numNeurons);
  m_B.randomize(1.0);

  m_W = Matrix(inputSize, numNeurons);
  m_W.randomize(1.0);
}

void DenseLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

std::array<size_t, 3> DenseLayer::outputSize() const {
  return { m_B.size(), 1, 1 };
}

const Vector& DenseLayer::activations() const {
  return m_A;
}

const Vector& DenseLayer::delta() const {
  return m_delta;
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

Vector DenseLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

void DenseLayer::trainForward(const Vector& inputs) {
  auto shouldDrop = [this]() {
    return rand() / (RAND_MAX + 1.0) < m_dropoutRate;
  };

  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);

  for (size_t a = 0; a < m_A.size(); ++a) {
    if (shouldDrop()) {
      m_A[a] = 0.0;
    }
  }
}

void DenseLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer, size_t epoch) {
  m_delta = nextLayer.W().transposeMultiply(nextLayer.delta())
                         .hadamard(m_Z.transform(sigmoidPrime));

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * learnRate;
}
