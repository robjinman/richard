#include <iostream> // TODO
#include <omp.h>
#include "max_pooling_layer.hpp"
#include "convolutional_layer.hpp"
#include "exception.hpp"

MaxPoolingLayer::MaxPoolingLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  size_t inputDepth)
  : m_Z(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth)
  , m_mask(inputW * inputH * inputDepth) {

  std::array<size_t, 2> regionSize = getOrThrow(obj, "regionSize").get<std::array<size_t, 2>>();
  m_regionW = regionSize[0];
  m_regionH = regionSize[1];

  TRUE_OR_THROW(inputW % m_regionW == 0,
    "Region width " << m_regionW << " does not divide input width " << inputW);
  TRUE_OR_THROW(inputH % m_regionH == 0,
    "Region height " << m_regionH << " does not divide input height " << inputH);
}

const Matrix& MaxPoolingLayer::W() const {
  assert(false);
  static Matrix m(1, 1);
  return m;
}

std::array<size_t, 3> MaxPoolingLayer::outputSize() const {
  return {
    static_cast<size_t>(m_inputW / m_regionW),
    static_cast<size_t>(m_inputH / m_regionH),
    m_inputDepth
  };
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
  m_Z = Vector(outputW * outputH * m_inputDepth);

  #pragma omp parallel for
  for (size_t slice = 0; slice < m_inputDepth; ++slice) {
    size_t inputOffset = slice * m_inputW * m_inputH;
    size_t outputOffset = slice * outputW * outputH;

    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        double largest = std::numeric_limits<double>::min();
        size_t largestInputX = 0;
        size_t largestInputY = 0;
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t inputX = x * m_regionW + i;
            size_t inputY = y * m_regionH + j;
            double input = inputs[inputOffset + inputY * m_inputW + inputX];
            if (input > largest) {
              largest = input;
              largestInputX = inputX;
              largestInputY = inputY;
            }
            m_mask[inputOffset + inputY * m_inputW + inputX] = 0.0;
          }
        }
        m_mask[inputOffset + largestInputY * m_inputW + largestInputX] = 1.0;
        m_Z[outputOffset + y * outputW + x] = largest;
      }
    }
  }

  //std::cout << "Mask\n";
  //std::cout << m_mask;
}

Vector MaxPoolingLayer::evalForward(const Vector& inputs) const {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;
  Vector Z(outputW * outputH * m_inputDepth);

  #pragma omp parallel for
  for (size_t slice = 0; slice < m_inputDepth; ++slice) {
    size_t inputOffset = slice * m_inputW * m_inputH;
    size_t outputOffset = slice * outputW * outputH;

    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        double largest = std::numeric_limits<double>::min();
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t inputX = x * m_regionW + i;
            size_t inputY = y * m_regionH + j;
            double input = inputs[inputOffset + inputY * m_inputW + inputX];
            if (input > largest) {
              largest = input;
            }
          }
        }
        Z[outputOffset + y * outputW + x] = largest;
      }
    }
  }

  return Z;
}

// Pad the delta to the input size using the mask for ease of consumption by the previous layer
void MaxPoolingLayer::padDelta(const Vector& delta) {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  m_delta = Vector(m_inputW * m_inputH * m_inputDepth);
  #pragma omp parallel for
  for (size_t slice = 0; slice < m_inputDepth; ++slice) {
    size_t inputOffset = slice * m_inputW * m_inputH;
    size_t outputOffset = slice * outputW * outputH;

    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t inputX = x * m_regionW + i;
            size_t inputY = y * m_regionH + j;
            if (m_mask[inputOffset + inputY * m_inputW + inputX] != 0.0) {
              m_delta[inputOffset + inputY * m_inputW + inputX] =
                delta[outputOffset + y * outputW + x];
            }
            else {
              m_delta[inputOffset + inputY * m_inputW + inputX] = 0.0;
            }
          }
        }
      }
    }
  }
}

void MaxPoolingLayer::backpropFromDenseLayer(const Layer& nextLayer) {
  Vector delta = nextLayer.W().transposeMultiply(nextLayer.delta());
  padDelta(delta);
}

void MaxPoolingLayer::backpropFromConvLayer(const Layer& nextLayer) {
  const auto& convLayer = dynamic_cast<const ConvolutionalLayer&>(nextLayer);
  auto kSz = convLayer.kernelSize();
  const Vector& convDelta = nextLayer.delta();
  size_t convLayerDepth = convLayer.depth();
  size_t kW = kSz[0];
  size_t kH = kSz[1];

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;
  size_t sliceSize = outputW * outputH;

  size_t fmW = outputW - kW + 1;
  size_t fmH = outputH - kH + 1;
  size_t fmSize = fmW * fmH;

  Vector delta(outputW * outputH * m_inputDepth);
  #pragma omp parallel for
  for (size_t slice = 0; slice < m_inputDepth; ++slice) {
    size_t sliceOffset = slice * sliceSize;

    for (size_t fm = 0; fm < convLayerDepth; ++fm) {
      const Matrix& kernel = convLayer.kernel(fm);
      size_t fmOffset = slice * convLayerDepth * fmSize + fm * fmSize;

      for (size_t fmY = 0; fmY < fmH; ++fmY) {
        for (size_t fmX = 0; fmX < fmW; ++fmX) {
          for (size_t j = 0; j < kH; ++j) {
            for (size_t i = 0; i < kW; ++i) {
              size_t x = fmX + i;
              size_t y = fmY + j;
              delta[sliceOffset + y * outputW + x] +=
                kernel.at(i, j) * convDelta[fmOffset + fmY * fmW + fmX];
            }
          }
        }
      }
    }
  }

  padDelta(delta);
}

void MaxPoolingLayer::updateDelta(const Vector&, const Layer& nextLayer, size_t) {
  switch (nextLayer.type()) {
    case LayerType::DENSE:
      backpropFromDenseLayer(nextLayer);
      break;
    case LayerType::CONVOLUTIONAL:
      backpropFromConvLayer(nextLayer);
      break;
    default:
      EXCEPTION("Expected layer of type DENSE or CONVOLUTIONAL, got " << nextLayer.type());
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
