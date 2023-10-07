#include <iostream> // TODO
#include <random>
#include "convolutional_layer.hpp"
#include "max_pooling_layer.hpp"
#include "exception.hpp"

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH)
  : m_W(1, 1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();

  m_W = Matrix(kernelSize[0], kernelSize[1]);
  m_W.randomize(1.0);

  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  m_b = dist(gen);
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH)
  : m_W(1, 1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();

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
      m_Z[ymin * featureMapW + xmin] = m_b;
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

  Vector Z(featureMapW * featureMapH);
  for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
    for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
      Z[ymin * featureMapW + xmin] = m_b;
      for (size_t j = 0; j < m_W.rows(); ++j) {
        for (size_t i = 0; i < m_W.cols(); ++i) {
          size_t inputX = xmin + i;
          size_t inputY = ymin + j;
          Z[ymin * featureMapW + xmin] += m_W.at(i, j) * inputs[inputY * m_inputW + inputX];
        }
      }
    }
  }

  return Z.transform(relu);
}

void ConvolutionalLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer,
  size_t epoch) {
  
  TRUE_OR_THROW(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  const MaxPoolingLayer& poolingLayer = dynamic_cast<const MaxPoolingLayer&>(nextLayer);
  const Vector& nextLayerDelta = poolingLayer.delta();

  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch) / (featureMapW * featureMapH);

  for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
    for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
      size_t idx = ymin * featureMapW + xmin;
      m_delta[idx] = nextLayerDelta[idx];
      for (size_t j = 0; j < m_W.rows(); ++j) {
        for (size_t i = 0; i < m_W.cols(); ++i) {
          size_t inputX = xmin + i;
          size_t inputY = ymin + j;
          double dw = layerInputs[inputY * m_inputW + inputX] * m_delta[idx] * learnRate;
          m_W.set(j, i, m_W.at(j, i) - dw);
          m_b = m_b - m_delta[idx] * learnRate;
        }
      }
    }
  }

  //std::cout << "Convolutional delta: \n";
  //std::cout << m_delta;
}

nlohmann::json ConvolutionalLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "convolutional";
  config["kernelSize"] = std::array<size_t, 2>({ m_W.cols(), m_W.rows() });
  config["learnRate"] = m_learnRate;
  config["learnRateDecay"] = m_learnRateDecay;
  return config;
}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(&m_b), sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

const Matrix& ConvolutionalLayer::W() const {
  return m_W;
}
