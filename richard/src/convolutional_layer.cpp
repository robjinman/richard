#include "convolutional_layer.hpp"
#include "max_pooling_layer.hpp"
#include "exception.hpp"

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  double learnRate)
  : m_W(1, 1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_learnRate(learnRate) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();

  m_W = Matrix(kernelSize[0], kernelSize[1]);

  //m_W = Matrix(width * height, 1);

  m_W.randomize(1.0);

  m_b = 0.0; // TODO: Randomize?
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH, double learnRate)
  : m_W(1, 1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_learnRate(learnRate) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();

  m_W = Matrix(kernelSize[0], kernelSize[1]);

  fin.read(reinterpret_cast<char*>(&m_b), sizeof(double));
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

const Vector& ConvolutionalLayer::activations() const {
  return m_A;
}

const Vector& ConvolutionalLayer::delta() const {
  return m_delta;
}

std::array<size_t, 2> ConvolutionalLayer::outputSize() const {
  return { m_inputW - m_W.cols(), m_inputH - m_W.rows() };
}

void ConvolutionalLayer::trainForward(const Vector& inputs) {
  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  m_Z = Vector(featureMapW * featureMapH);
  m_A = Vector(featureMapW * featureMapH);
  m_delta = Vector(featureMapW * featureMapH);
  for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
    for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
      m_Z[ymin * featureMapW + xmin] = 0.0;
      for (size_t j = 0; j < m_W.rows(); ++j) {
        for (size_t i = 0; i < m_W.cols(); ++i) {
          size_t inputX = xmin + i;
          size_t inputY = ymin + j;
          m_Z[ymin * featureMapW + xmin] += m_W.at(i, j) * inputs[inputY * m_inputW + inputX];
        }
      }
    }
  }

  m_A = m_Z.transform(relu);
}

Vector ConvolutionalLayer::evalForward(const Vector& inputs) const {
  size_t featureMapW = m_inputW - m_W.cols();
  size_t featureMapH = m_inputH - m_W.rows();

  Vector A(featureMapW * featureMapH);
  for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
    for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
      for (size_t j = 0; j < m_W.rows(); ++j) {
        for (size_t i = 0; i < m_W.cols(); ++i) {
          size_t inputX = xmin + i;
          size_t inputY = ymin + j;
          A[ymin * featureMapW + xmin] += relu(m_W.at(i, j) * inputs[inputY * m_inputW + inputX]);
        }
      }
    }
  }

  return A;
}

void ConvolutionalLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {
  TRUE_OR_THROW(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  const MaxPoolingLayer& poolingLayer = dynamic_cast<const MaxPoolingLayer&>(nextLayer);
  const Vector& nextLayerDelta = poolingLayer.delta();

  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  double learnRate = m_learnRate / (featureMapW * featureMapH);

  for (size_t j = 0; j < featureMapH; ++j) {
    for (size_t i = 0; i < featureMapW; ++i) {
      size_t idx = j * featureMapW + i;
      m_delta[idx] = nextLayerDelta[idx];
      for (size_t b = 0; b < m_W.rows(); ++b) {
        for (size_t a = 0; a < m_W.cols(); ++a) {
          size_t x = i + a;
          size_t y = j + b;
          double dw = layerInputs[y * m_inputW + x] * m_delta[idx] * learnRate;
          m_W.set(b, a, m_W.at(b, a) - dw);
        }
      }
      m_b = m_b - m_delta[idx] * learnRate;
    }
  }

  //std::cout << "Convolutional delta: \n";
  //std::cout << m_delta;
}

nlohmann::json ConvolutionalLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "convolutional";
  config["kernelSize"] = std::array<size_t, 2>({ m_W.cols(), m_W.rows() });
  return config;
}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(&m_b), sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

const Matrix& ConvolutionalLayer::W() const {
  return m_W;
}
