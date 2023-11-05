#include <random>
#include "convolutional_layer.hpp"
#include "max_pooling_layer.hpp"
#include "exception.hpp"
#include "util.hpp"

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  size_t inputDepth)
  : m_Z(1, 1, 1)
  , m_A(1, 1, 1)
  , m_delta(1, 1, 1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    Filter filter;

    filter.K = Kernel(kernelSize[0], kernelSize[1], inputDepth);
    filter.K.randomize(0.1);

    filter.b = 0.0;

    m_filters.push_back(filter);
  }

  auto sz = outputSize();

  m_Z = Array3(sz[0], sz[1], sz[2]);
  m_A = Array3(sz[0], sz[1], sz[2]);
  m_delta = Array3(sz[0], sz[1], sz[2]);
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH, size_t inputDepth)
  : m_Z(1, 1, 1)
  , m_A(1, 1, 1)
  , m_delta(1, 1, 1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    Filter filter;

    filter.K = Kernel(kernelSize[0], kernelSize[1], inputDepth);

    fin.read(reinterpret_cast<char*>(&filter.b), sizeof(double));
    fin.read(reinterpret_cast<char*>(filter.K.data()),
      filter.K.W() * filter.K.H() * filter.K.D() * sizeof(double));

    m_filters.push_back(filter);
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

Triple ConvolutionalLayer::outputSize() const {
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
    double b = m_filters[slice].b;

    K.convolve(inputs, *featureMap);

    (*featureMap) += b;
  }
}

void ConvolutionalLayer::trainForward(const DataArray& inputs) {
  ConstArray3Ptr pX = Array3::createShallow(inputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& X = *pX;

  forwardPass(X, m_Z);

  m_A = m_Z.computeTransform(relu);
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

void ConvolutionalLayer::updateDelta(const DataArray& layerInputs, const Layer& nextLayer,
  size_t epoch) {

  ASSERT_MSG(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  size_t fmW = outputSize()[0];
  size_t fmH = outputSize()[1];
  size_t depth = m_filters.size();

  ConstArray3Ptr pNextDelta = Array3::createShallow(nextLayer.delta(), fmW, fmH, depth);
  const Array3& nextDelta = *pNextDelta;

  ConstArray3Ptr pInputs = Array3::createShallow(layerInputs, m_inputW, m_inputH, m_inputDepth);
  const Array3& inputs = *pInputs;

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch) / (fmW * fmH);

  for (size_t slice = 0; slice < depth; ++slice) {
    Kernel& K = m_filters[slice].K;
    double& b = m_filters[slice].b;

    for (size_t ymin = 0; ymin < fmH; ++ymin) {
      for (size_t xmin = 0; xmin < fmW; ++xmin) {
        double delta = reluPrime(m_Z.at(xmin, ymin, slice)) * nextDelta.at(xmin, ymin, slice);
        m_delta.set(xmin, ymin, slice, delta);

        for (size_t z = 0; z < K.D(); ++z) {
          for (size_t j = 0; j < K.H(); ++j) {
            for (size_t i = 0; i < K.W(); ++i) {
              size_t inputX = xmin + i;
              size_t inputY = ymin + j;
              double dw = inputs.at(inputX, inputY, z) * delta * learnRate;

              K.set(i, j, z, K.at(i, j, z) - dw);
              b = b - delta * learnRate;
            }
          }
        }
      }
    }
  }
}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {
  for (const Filter& filter : m_filters) {
    fout.write(reinterpret_cast<const char*>(&filter.b), sizeof(double));
    fout.write(reinterpret_cast<const char*>(filter.K.data()), filter.K.size() * sizeof(double));
  }
}

const Matrix& ConvolutionalLayer::W() const {
  assert(false);
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

void ConvolutionalLayer::setFilters(const std::vector<ConvolutionalLayer::Filter>& filters) {
  ASSERT(filters.size() == m_filters.size());
  m_filters = filters;
}

void ConvolutionalLayer::setWeights(const std::vector<DataArray>& weights) {
  ASSERT(weights.size() == m_filters.size());

  for (size_t i = 0; i < m_filters.size(); ++i) {
    const Kernel& K = m_filters[i].K;
    m_filters[i].K = Kernel(weights[i], K.W(), K.H(), K.D());
  }
}

void ConvolutionalLayer::setBiases(const DataArray& biases) {
  ASSERT(biases.size() == m_filters.size());
  ConstVectorPtr pB = Vector::createShallow(biases);
  const Vector& B = *pB;
  
  for (size_t i = 0; i < m_filters.size(); ++i) {
    m_filters[i].b = B[i];
  }
}

