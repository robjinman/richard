#include "cpu/max_pooling_layer.hpp"
#include "exception.hpp"
#include "utils.hpp"

namespace richard {
namespace cpu {

MaxPoolingLayer::MaxPoolingLayer(const nlohmann::json& obj, const Size3& inputShape)
  : m_inputDelta(inputShape[0], inputShape[1], inputShape[2])
  , m_inputW(inputShape[0])
  , m_inputH(inputShape[1])
  , m_inputDepth(inputShape[2])
  , m_mask(m_inputW, m_inputH, m_inputDepth) {

  std::array<size_t, 2> regionSize = getOrThrow(obj, "regionSize").get<std::array<size_t, 2>>();
  m_regionW = regionSize[0];
  m_regionH = regionSize[1];

  ASSERT_MSG(m_inputW % m_regionW == 0,
    "Region width " << m_regionW << " does not divide input width " << m_inputW);
  ASSERT_MSG(m_inputH % m_regionH == 0,
    "Region height " << m_regionH << " does not divide input height " << m_inputH);

  m_Z = Array3(m_inputW / m_regionW, m_inputH / m_regionH, m_inputDepth);
}

Size3 MaxPoolingLayer::outputSize() const {
  return {
    static_cast<size_t>(m_inputW / m_regionW),
    static_cast<size_t>(m_inputH / m_regionH),
    m_inputDepth
  };
}

const DataArray& MaxPoolingLayer::activations() const {
  return m_Z.storage();
}

const DataArray& MaxPoolingLayer::inputDelta() const {
  return m_inputDelta.storage();
}

void MaxPoolingLayer::trainForward(const DataArray& inputs) {
  ConstArray3Ptr pImage = Array3::createShallow(inputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& image = *pImage;

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  for (size_t z = 0; z < m_inputDepth; ++z) {
    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        netfloat_t largest = std::numeric_limits<netfloat_t>::lowest();
        size_t largestInputX = 0;
        size_t largestInputY = 0;

        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t imgX = x * m_regionW + i;
            size_t imgY = y * m_regionH + j;
            netfloat_t input = image.at(imgX, imgY, z);

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
        netfloat_t largest = std::numeric_limits<netfloat_t>::lowest();

        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t imgX = x * m_regionW + i;
            size_t imgY = y * m_regionH + j;
            netfloat_t input = image.at(imgX, imgY, z);

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

void MaxPoolingLayer::updateDeltas(const DataArray&, const DataArray& outputDelta) {
  ConstArray3Ptr pDelta = Array3::createShallow(outputDelta, m_Z.W(), m_Z.H(), m_Z.D());
  const Array3& delta = *pDelta;

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  for (size_t z = 0; z < m_inputDepth; ++z) {
    for (size_t y = 0; y < outputH; ++y) {
      for (size_t x = 0; x < outputW; ++x) {
        for (size_t j = 0; j < m_regionH; ++j) {
          for (size_t i = 0; i < m_regionW; ++i) {
            size_t imgX = x * m_regionW + i;
            size_t imgY = y * m_regionH + j;

            if (m_mask.at(imgX, imgY, z) != 0.0) {
              m_inputDelta.set(imgX, imgY, z, delta.at(x, y, z));
            }
            else {
              m_inputDelta.set(imgX, imgY, z, 0.0);
            }
          }
        }
      }
    }
  }
}

const Array3& MaxPoolingLayer::test_mask() const {
  return m_mask;
}

}
}
