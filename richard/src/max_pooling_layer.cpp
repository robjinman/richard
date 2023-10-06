#include <iostream> // TODO
#include "max_pooling_layer.hpp"
#include "exception.hpp"

MaxPoolingLayer::MaxPoolingLayer(const nlohmann::json& obj, size_t inputW, size_t inputH)
  : m_Z(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_mask(inputW * inputH) {

  std::array<size_t, 2> regionSize = getOrThrow(obj, "regionSize").get<std::array<size_t, 2>>();
  m_regionW = regionSize[0];
  m_regionH = regionSize[1];

  TRUE_OR_THROW(inputW % m_regionW == 0, "region width does not divide input width");
  TRUE_OR_THROW(inputH % m_regionH == 0, "region height does not divide input height");
}

const Matrix& MaxPoolingLayer::W() const {
  assert(false);
  static Matrix m(1, 1);
  return m;
}

std::array<size_t, 2> MaxPoolingLayer::outputSize() const {
  return { static_cast<size_t>(m_inputW / m_regionW), static_cast<size_t>(m_inputH / m_regionH) };
}

const Vector& MaxPoolingLayer::activations() const {
  return m_Z;
}

const Vector& MaxPoolingLayer::delta() const {
  return m_delta;
}

void MaxPoolingLayer::trainForward(const Vector& inputs) {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;
  m_Z = Vector(outputW * outputH);

  //std::cout << "outputW: " << outputW << "\n";
  //std::cout << "outputH: " << outputH << "\n";
  //std::cout << "m_regionW: " << m_regionW << "\n";
  //std::cout << "m_regionH: " << m_regionH << "\n";

  for (size_t y = 0; y < outputH; ++y) {
    for (size_t x = 0; x < outputW; ++x) {
      double largest = std::numeric_limits<double>::min();
      size_t largestInputX = 0;
      size_t largestInputY = 0;
      for (size_t j = 0; j < m_regionH; ++j) {
        for (size_t i = 0; i < m_regionW; ++i) {
          size_t inputX = x * m_regionW + i;
          size_t inputY = y * m_regionH + j;
          double input = inputs[inputY * m_inputW + inputX];
          if (input > largest) {
            largest = input;
            largestInputX = inputX;
            largestInputY = inputY;
          }
          m_mask[inputY * m_inputW + inputX] = 0.0;
        }
      }
      m_mask[largestInputY * m_inputW + largestInputX] = 1.0;
      m_Z[y * outputW + x] = largest;
    }
  }

  //std::cout << "Mask\n";
  //std::cout << m_mask;
}

Vector MaxPoolingLayer::evalForward(const Vector& inputs) const {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;
  Vector Z(outputW * outputH);

  for (size_t y = 0; y < outputH; ++y) {
    for (size_t x = 0; x < outputW; ++x) {
      double largest = std::numeric_limits<double>::min();
      for (size_t j = 0; j < m_regionH; ++j) {
        for (size_t i = 0; i < m_regionW; ++i) {
          size_t inputX = x * m_regionW + i;
          size_t inputY = y * m_regionH + j;
          double input = inputs[inputY * m_inputW + inputX];
          if (input > largest) {
            largest = input;
          }
        }
      }
      Z[y * outputW + x] = largest;
    }
  }

  return Z;
}

void MaxPoolingLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {
  m_delta = Vector(m_inputW * m_inputH);

  Vector delta = nextLayer.W().transposeMultiply(nextLayer.delta());

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  for (size_t y = 0; y < outputH; ++y) {
    for (size_t x = 0; x < outputW; ++x) {
      for (size_t j = 0; j < m_regionH; ++j) {
        for (size_t i = 0; i < m_regionW; ++i) {
          size_t inputX = x * m_regionW + i;
          size_t inputY = y * m_regionH + j;
          if (m_mask[inputY * m_inputW + inputX] != 0.0) {
            m_delta[inputY * m_inputW + inputX] = delta[y * outputW + x];
          }
          else {
            m_delta[inputY * m_inputW + inputX] = 0.0;
          }
        }
      }
    }
  }

  //std::cout << "Max pooling delta: \n";
  //std::cout << m_delta;
}

nlohmann::json MaxPoolingLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "maxPooling";
  config["regionSize"] = std::array<size_t, 2>({ m_regionW, m_regionH });
  return config;
}
