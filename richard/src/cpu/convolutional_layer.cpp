#include "cpu/convolutional_layer.hpp"
#include "cpu/max_pooling_layer.hpp"
#include "exception.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <random>

namespace richard {
namespace cpu {

ConvolutionalLayer::ConvolutionalLayer(const Config& config, const Size3& inputShape) {
  initialize(config, inputShape);
}

ConvolutionalLayer::ConvolutionalLayer(const Config& config, std::istream& stream,
  const Size3& inputShape) {

  initialize(config, inputShape);

  for (Filter& filter : m_filters) {
    stream.read(reinterpret_cast<char*>(&filter.b), sizeof(netfloat_t));
    stream.read(reinterpret_cast<char*>(filter.K.data()),
      filter.K.W() * filter.K.H() * filter.K.D() * sizeof(netfloat_t));
  }
}

void ConvolutionalLayer::initialize(const Config& config, const Size3& inputShape) {
  m_inputW = inputShape[0];
  m_inputH = inputShape[1];
  m_inputDepth = inputShape[2];

  auto kernelSize = config.getNumberArray<size_t, 2>("kernelSize");
  m_learnRate = config.getNumber<netfloat_t>("learnRate");
  m_learnRateDecay = config.getNumber<netfloat_t>("learnRateDecay");
  size_t depth = config.getNumber<size_t>("depth");
  m_dropoutRate = config.getNumber<netfloat_t>("dropoutRate");

  ASSERT_MSG(kernelSize[0] <= m_inputW,
    "Kernel width " << kernelSize[0] << " is larger than input width " << m_inputW);

  ASSERT_MSG(kernelSize[1] <= m_inputH,
    "Kernel height " << kernelSize[1] << " is larger than input height " << m_inputH);

  for (size_t i = 0; i < depth; ++i) {
    m_filters.push_back(Filter{ Kernel(kernelSize[0], kernelSize[1], m_inputDepth), 0.f });
    m_filters.back().K.randomize(0.1f);

    m_paramDeltas.push_back(Filter{ Kernel(kernelSize[0], kernelSize[1], m_inputDepth), 0.f });
  }

  auto sz = outputSize();
  m_Z = Array3(sz[0], sz[1], sz[2]);
  m_A = Array3(sz[0], sz[1], sz[2]);
  m_inputDelta = Array3(m_inputW, m_inputH, m_inputDepth);
}

const DataArray& ConvolutionalLayer::activations() const {
  return m_A.storage();
}

const DataArray& ConvolutionalLayer::inputDelta() const {
  return m_inputDelta.storage();
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

    computeCrossCorrelation(inputs, K, *featureMap);

    *featureMap += b;
  }
}

void ConvolutionalLayer::trainForward(const DataArray& inputs) {
  auto shouldDrop = [this]() {
    return rand() / (RAND_MAX + 1.0) < m_dropoutRate;
  };
  
  auto reluWithDropout = [&](netfloat_t x) -> netfloat_t {
    return shouldDrop() ? 0.f : relu(x); 
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

void ConvolutionalLayer::updateDeltas(const DataArray& layerInputs, const DataArray& outputDelta) {
  size_t fmW = outputSize()[0];
  size_t fmH = outputSize()[1];
  size_t depth = m_filters.size();

  ConstArray3Ptr pDeltaA = Array3::createShallow(outputDelta, fmW, fmH, depth);
  const Array3& deltaA = *pDeltaA;

  ConstArray3Ptr pInputs3 = Array3::createShallow(layerInputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& inputs3 = *pInputs3;

  Array3 delta3 = deltaA.hadamard(m_Z.computeTransform(reluPrime));
  m_inputDelta.zero();

  Array2 dInputDelta(m_inputDelta.W(), m_inputDelta.H());
  DBG_ASSERT(m_filters.size() > 0);
  Array2 dDeltaK(m_filters[0].K.W(), m_filters[0].K.H());

  for (size_t slice = 0; slice < depth; ++slice) {
    const Kernel& K = m_filters[slice].K;
    Kernel& deltaK3 = m_paramDeltas[slice].K;
    netfloat_t& db = m_paramDeltas[slice].b;

    ConstArray2Ptr pDelta = delta3.slice(slice);
    const Array2& delta = *pDelta;

    for (size_t z = 0; z < K.D(); ++z) {
      ConstArray2Ptr pW = K.slice(z);
      const Array2& W = *pW;

      Array2Ptr pInputDelta = m_inputDelta.slice(z);
      Array2& inputDelta = *pInputDelta;

      computeFullConvolution(W, delta, dInputDelta);

      inputDelta += dInputDelta;

      ConstArray2Ptr pInputs = inputs3.slice(z);
      const Array2& inputs = *pInputs;

      Array2Ptr pDeltaK = deltaK3.slice(z);
      Array2& deltaK = *pDeltaK;

      computeCrossCorrelation(inputs, delta, dDeltaK);
      deltaK += dDeltaK;
    }

    db += delta.sum();
  }
}

void ConvolutionalLayer::updateParams(size_t epoch) {
  netfloat_t learnRate = m_learnRate * static_cast<netfloat_t>(pow(m_learnRateDecay, epoch));

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

void ConvolutionalLayer::test_setFilters(const std::vector<Filter>& filters) {
  m_filters = filters;
}

const std::vector<ConvolutionalLayer::Filter> ConvolutionalLayer::test_filters() const {
  return m_filters;
}

const std::vector<ConvolutionalLayer::Filter> ConvolutionalLayer::test_filterDeltas() const {
  return m_paramDeltas;
}

}
}
