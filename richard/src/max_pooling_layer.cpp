#include "max_pooling_layer.hpp"
#include "convolutional_layer.hpp"
#include "exception.hpp"

MaxPoolingLayer::MaxPoolingLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  size_t inputDepth)
  : m_Z(1, 1, 1)
  , m_delta(1, 1, 1)
  , m_paddedDelta(inputW, inputH, inputDepth)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth)
  , m_mask(inputW, inputH, inputDepth) {

  std::array<size_t, 2> regionSize = getOrThrow(obj, "regionSize").get<std::array<size_t, 2>>();
  m_regionW = regionSize[0];
  m_regionH = regionSize[1];

  TRUE_OR_THROW(inputW % m_regionW == 0,
    "Region width " << m_regionW << " does not divide input width " << inputW);
  TRUE_OR_THROW(inputH % m_regionH == 0,
    "Region height " << m_regionH << " does not divide input height " << inputH);

  m_Z = Array3(m_inputW / m_regionW, m_inputH / m_regionH, m_inputDepth);
  m_delta = Array3(m_inputW / m_regionW, m_inputH / m_regionH, m_inputDepth);
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

const DataArray& MaxPoolingLayer::activations() const {
  return m_Z.storage();
}

const DataArray& MaxPoolingLayer::delta() const {
  return m_paddedDelta.storage();
}

void MaxPoolingLayer::trainForward(const DataArray& inputs) {
  ConstArray3Ptr pImage = Array3::createShallow(inputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& image = *pImage;

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  for (size_t z = 0; z < m_inputDepth; ++z) {
    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        double largest = std::numeric_limits<double>::min();
        size_t largestInputX = 0;
        size_t largestInputY = 0;
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t imgX = x * m_regionW + i;
            size_t imgY = y * m_regionH + j;
            double input = image.at(imgX, imgY, z);
            if (input > largest) {
              largest = input;
              largestInputX = imgX;
              largestInputY = imgY;
            }
            m_mask.set(imgX, imgY, z, 0.0);
          }
        }
        m_mask.set(largestInputX, largestInputY, z, 1.0);
        m_Z.set(x, y, z, largest);
      }
    }
  }
}

DataArray MaxPoolingLayer::evalForward(const DataArray& inputs) const {
  ConstArray3Ptr pImage = Array3::createShallow(inputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& image = *pImage;

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  Array3 Z(outputW, outputH, m_inputDepth);

  for (size_t z = 0; z < m_inputDepth; ++z) {
    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        double largest = std::numeric_limits<double>::min();
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t imgX = x * m_regionW + i;
            size_t imgY = y * m_regionH + j;
            double input = image.at(imgX, imgY, z);
            if (input > largest) {
              largest = input;
            }
          }
        }
        Z.set(x, y, z, largest);
      }
    }
  }

  return Z.storage();
}

// Pad the delta to the input size using the mask for ease of consumption by the previous layer
void MaxPoolingLayer::padDelta(const Array3& delta, const Array3& mask, Array3& paddedDelta) const {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  for (size_t z = 0; z < m_inputDepth; ++z) {
    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t imgX = x * m_regionW + i;
            size_t imgY = y * m_regionH + j;
            if (mask.at(imgX, imgY, z) != 0.0) {
              paddedDelta.set(imgX, imgY, z, delta.at(x, y, z));
            }
            else {
              paddedDelta.set(imgX, imgY, z, 0.0);
            }
          }
        }
      }
    }
  }
}

void MaxPoolingLayer::backpropFromDenseLayer(const Layer& nextLayer, Vector& delta) {
  ConstVectorPtr pNextDelta = Vector::createShallow(nextLayer.delta());
  delta = nextLayer.W().transposeMultiply(*pNextDelta);
}

void MaxPoolingLayer::backpropFromConvLayer(const std::vector<ConvolutionalLayer::Filter>& filters,
  const DataArray& convDelta, Array3& delta) {

  size_t convLayerDepth = filters.size();
  ASSERT(convLayerDepth > 0);
  size_t kW = filters[0].K.W();
  size_t kH = filters[0].K.H();
  size_t kD = filters[0].K.D();
  ASSERT(kD == m_inputDepth);

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  size_t fmW = outputW - kW + 1;
  size_t fmH = outputH - kH + 1;

  ConstArray3Ptr pNextDelta = Array3::createShallow(convDelta, fmW, fmH, convLayerDepth);
  const Array3& nextDelta = *pNextDelta;

  for (size_t fm = 0; fm < convLayerDepth; ++fm) {
    const Kernel& kernel = filters[fm].K;

    for (size_t fmY = 0; fmY < fmH; ++fmY) {
      for (size_t fmX = 0; fmX < fmW; ++fmX) {
        for (size_t z = 0; z < kD; ++z) {
          for (size_t j = 0; j < kH; ++j) {
            for (size_t i = 0; i < kW; ++i) {
              size_t x = fmX + i;
              size_t y = fmY + j;
              double d = kernel.at(i, j, z) * nextDelta.at(fmX, fmY, fm);
              delta.set(x, y, z, delta.at(x, y, z) + d);
            }
          }
        }
      }
    }
  }
}

void MaxPoolingLayer::updateDelta(const DataArray&, const Layer& nextLayer, size_t) {
  switch (nextLayer.type()) {
    case LayerType::OUTPUT:
    case LayerType::DENSE: {
      VectorPtr pDeltaVec = Vector::createShallow(m_delta.storage());
      backpropFromDenseLayer(nextLayer, *pDeltaVec);
      break;
    }
    case LayerType::CONVOLUTIONAL: {
      const auto& convLayer = dynamic_cast<const ConvolutionalLayer&>(nextLayer);
      backpropFromConvLayer(convLayer.filters(), convLayer.delta(), m_delta);
      break;
    }
    default: {
      EXCEPTION("Expected layer of type DENSE or CONVOLUTIONAL, got " << nextLayer.type());
    }
  }

  padDelta(m_delta, m_mask, m_paddedDelta);
}

nlohmann::json MaxPoolingLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "maxPooling";
  config["regionSize"] = std::array<size_t, 2>({ m_regionW, m_regionH });
  return config;
}

const Array3& MaxPoolingLayer::mask() const {
  return m_mask;
}
