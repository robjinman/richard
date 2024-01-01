#include "cpu/convolutional_layer.hpp"
#include "cpu/max_pooling_layer.hpp"
#include "exception.hpp"
#include "utils.hpp"
#include <random>

namespace richard {
namespace cpu {

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, const Size3& inputShape) {
  initialize(obj, inputShape);
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& stream,
  const Size3& inputShape) {

  initialize(obj, inputShape);

  for (Filter& filter : m_filters) {
    stream.read(reinterpret_cast<char*>(&filter.b), sizeof(netfloat_t));
    stream.read(reinterpret_cast<char*>(filter.K.data()),
      filter.K.W() * filter.K.H() * filter.K.D() * sizeof(netfloat_t));
  }
}

void ConvolutionalLayer::initialize(const nlohmann::json& obj, const Size3& inputShape) {
  m_inputW = inputShape[0];
  m_inputH = inputShape[1];
  m_inputDepth = inputShape[2];

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<netfloat_t>();

  for (size_t i = 0; i < depth; ++i) {
    m_filters.push_back(Filter{ Kernel(kernelSize[0], kernelSize[1], m_inputDepth), 0.f });
    m_filters.back().K.randomize(0.1);

    m_paramDeltas.push_back(Filter{ Kernel(kernelSize[0], kernelSize[1], m_inputDepth), 0.f });
  }

  auto sz = outputSize();
  m_Z = Array3(sz[0], sz[1], sz[2]);
  m_A = Array3(sz[0], sz[1], sz[2]);
  m_delta = Array3(sz[0], sz[1], sz[2]);
}

const DataArray& ConvolutionalLayer::activations() const {
  return m_A.storage();
}

const DataArray& ConvolutionalLayer::delta() const {
  return m_delta.storage();
}

Size3 ConvolutionalLayer::outputSize() const {
  DBG_ASSERT(!m_filters.empty());
  return {
    m_inputW - m_filters[0].K.W() + 1,
    m_inputH - m_filters[0].K.H() + 1,
    m_filters.size()
  };
}

size_t ConvolutionalLayer::numOutputs() const {
  auto os = outputSize();
  return os[0] * os[1] * os[2];
}

void ConvolutionalLayer::forwardPass(const Array3& inputs, Array3& Z) const {
  size_t depth = m_filters.size();

  for (size_t slice = 0; slice < depth; ++slice) {
    Array2Ptr featureMap = Z.slice(slice);

    const Kernel& K = m_filters[slice].K;
    netfloat_t b = m_filters[slice].b;

    K.convolve(inputs, *featureMap);

    (*featureMap) += b;
  }
}

void ConvolutionalLayer::trainForward(const DataArray& inputs) {
  auto shouldDrop = [this]() {
    return rand() / (RAND_MAX + 1.0) < m_dropoutRate;
  };
  
  auto reluWithDropout = [&](netfloat_t x) {
    return shouldDrop() ? 0.0 : relu(x); 
  };

  ConstArray3Ptr pX = Array3::createShallow(inputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& X = *pX;

  forwardPass(X, m_Z);

  m_A = m_Z.computeTransform(reluWithDropout);
}

DataArray ConvolutionalLayer::evalForward(const DataArray& inputs) const {
  ConstArray3Ptr pX = Array3::createShallow(inputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& X = *pX;

  auto sz = outputSize();
  Array3 Z(sz[0], sz[1], sz[2]);

  forwardPass(X, Z);

  Z.transformInPlace(relu);

  return Z.storage();
}

void ConvolutionalLayer::updateDelta(const DataArray& layerInputs, const Layer& nextLayer) {
  ASSERT_MSG(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  size_t fmW = outputSize()[0];
  size_t fmH = outputSize()[1];
  size_t depth = m_filters.size();

  ConstArray3Ptr pNextDelta = Array3::createShallow(nextLayer.delta(), fmW, fmH, depth);
  const Array3& nextDelta = *pNextDelta;

  ConstArray3Ptr pInputs = Array3::createShallow(layerInputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& inputs = *pInputs;

  for (size_t slice = 0; slice < depth; ++slice) {
    Kernel& dK = m_paramDeltas[slice].K;
    netfloat_t& db = m_paramDeltas[slice].b;

    for (size_t ymin = 0; ymin < fmH; ++ymin) {
      for (size_t xmin = 0; xmin < fmW; ++xmin) {
        netfloat_t delta = reluPrime(m_Z.at(xmin, ymin, slice)) * nextDelta.at(xmin, ymin, slice);
        m_delta.set(xmin, ymin, slice, delta);
      }
    }

    for (size_t z = 0; z < dK.D(); ++z) {
      for (size_t j = 0; j < dK.H(); ++j) {
        for (size_t i = 0; i < dK.W(); ++i) {

          netfloat_t sum = 0.0;

          for (size_t ymin = 0; ymin < fmH; ++ymin) {
            for (size_t xmin = 0; xmin < fmW; ++xmin) {
              size_t inputX = xmin + i;
              size_t inputY = ymin + j;

              netfloat_t delta = m_delta.at(xmin, ymin, slice);

              sum += inputs.at(inputX, inputY, z) * delta;
              db += delta;
            }
          }

          dK.set(i, j, z, dK.at(i, j, z) + sum);
        }
      }
    }
  }
}

void ConvolutionalLayer::updateParams(size_t epoch) {
  size_t featureMapSize = outputSize()[0] * outputSize()[1];
  netfloat_t learnRate = m_learnRate * pow(m_learnRateDecay, epoch) / featureMapSize;

  for (size_t slice = 0; slice < m_filters.size(); ++slice) {
    m_filters[slice].K -= m_paramDeltas[slice].K * learnRate;
    m_filters[slice].b -= m_paramDeltas[slice].b * learnRate;

    m_paramDeltas[slice].K.zero();
    m_paramDeltas[slice].b = 0.0;
  }
}

void ConvolutionalLayer::writeToStream(std::ostream& stream) const {
  for (const Filter& filter : m_filters) {
    stream.write(reinterpret_cast<const char*>(&filter.b), sizeof(netfloat_t));
    stream.write(reinterpret_cast<const char*>(filter.K.data()),
      filter.K.size() * sizeof(netfloat_t));
  }
}

const Matrix& ConvolutionalLayer::W() const {
  EXCEPTION("Use ConvolutionalLayer::filters() instead of ConvolutionalLayer::W()");
  static Matrix m(1, 1);
  return m;
}

size_t ConvolutionalLayer::depth() const {
  return m_filters.size();
}

const std::vector<ConvolutionalLayer::Filter>& ConvolutionalLayer::filters() const {
  return m_filters;
}

std::array<size_t, 2> ConvolutionalLayer::kernelSize() const {
  ASSERT(m_filters.size() > 0);
  return { m_filters[0].K.W(), m_filters[0].K.H() };
}

void ConvolutionalLayer::test_setFilters(const std::vector<ConvolutionalLayer::Filter>& filters) {
  ASSERT(filters.size() == m_filters.size());
  m_filters = filters;
}

void ConvolutionalLayer::test_forwardPass(const Array3& inputs, Array3& Z) const {
  forwardPass(inputs, Z);
}

}
}
